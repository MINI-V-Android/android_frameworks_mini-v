package com.miniv.ai.engine;

import android.os.IBinder;
import android.os.RemoteException;
import android.os.ServiceManager;
import android.util.Log;

import vendor.miniv.ai.IMiniVAiHal;
import vendor.miniv.ai.IMiniVAiStreamCallback;

import java.util.Collections;
import java.util.HashSet;
import java.util.Set;

/**
 * LLM Engine implementation backed by the vendor NPU HAL
 * (vendor.miniv.ai.IMiniVAiHal).
 *
 * Bridges LLMEngine's common taxonomy to the vendor HAL's AIDL interface.
 * Error code mapping is intentionally 1:1 by design (see IMiniVAiHal.aidl /
 * IMiniVAiStreamCallback.aidl comments, "Phase 3.4"):
 *   IMiniVAiStreamCallback -50x  ->  LLMEngine.ErrorCode -100x
 */
public class VendorNpuLLMEngine implements LLMEngine {
    private static final String TAG = "VendorNpuLLMEngine";
    private static final String HAL_INSTANCE = "vendor.miniv.ai.IMiniVAiHal/default";

    // NOTE: the vendor daemon (NpuLLMEngine::infer()) does zero prompt
    // formatting — it tokenizes exactly whatever string it's given. Every
    // caller of IMiniVAiHal.inferStream() is responsible for building the
    // model's expected chat format itself. Qwen2.5 uses ChatML; this is
    // hardcoded here because IMiniVAiHal doesn't currently expose the
    // loaded model's chat template. If/when multi-model support lands,
    // this needs to move behind something like hal.getChatTemplate().
    private static final String DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant.";
    
    private final Set<Integer> mSeenSessions =
            Collections.synchronizedSet(new HashSet<>());

    private String applyChatTemplate(int sessionId, String userPrompt) {
        boolean firstTurn = mSeenSessions.add(sessionId); 

        if (firstTurn) {
            return "<|im_start|>system\n" + DEFAULT_SYSTEM_PROMPT + "<|im_end|>\n"
                    + "<|im_start|>user\n" + userPrompt + "<|im_end|>\n"
                    + "<|im_start|>assistant\n";
        }
        return "<|im_end|>\n<|im_start|>user\n" + userPrompt + "<|im_end|>\n"
                + "<|im_start|>assistant\n";
    }

    private static final int MODEL_N_CTX = 2048;
    private static final int CONTEXT_SAFETY_MARGIN = 64;

    /**
     * Clamp maxTokens so that (estimated prompt tokens + maxTokens) stays
     * within the model's fixed context window. NpuLLMEngine::infer() does
     * not check this itself — nothing downstream does — so without this,
     * a caller requesting a large maxTokens on a long prompt can overrun
     * the KV cache with undefined behavior.
     *
     * Token count is approximated conservatively (~2 chars/token, safe for
     * mixed Korean/English/CJK input where real tokenizers tend to produce
     * *more* tokens per char than plain English) since no real tokenize()
     * call is available here.
     */
    private static int clampMaxTokens(String formattedPrompt, int requestedMaxTokens) {
        int estimatedPromptTokens = (formattedPrompt.length() / 2) + 1;
        int budget = MODEL_N_CTX - estimatedPromptTokens - CONTEXT_SAFETY_MARGIN;
        if (budget < 1) {
            budget = 1; // still attempt something rather than refusing outright
        }
        return Math.min(requestedMaxTokens, budget);
    }

    // Vendor-HAL-internal error codes reported via IMiniVAiStreamCallback.onError()
    // and IMiniVAiHal's synchronous return values — kept private to this class,
    // translated to LLMEngine.ErrorCode before ever reaching TokenCallback.
    private interface InternalErrorCode {
        int SESSION_NOT_FOUND = IMiniVAiStreamCallback.ERROR_SESSION_NOT_FOUND;       // -501
        int CACHE_LIMIT_EXCEEDED = IMiniVAiStreamCallback.ERROR_CACHE_LIMIT_EXCEEDED; // -502
        int GENERIC_FAILURE = IMiniVAiStreamCallback.ERROR_GENERIC_FAILURE;           // -503
    }

    private volatile IMiniVAiHal mHal;

    public VendorNpuLLMEngine() {
        mHal = null;
    }

    private static IMiniVAiHal fetchHal() {
        IBinder binder = ServiceManager.waitForDeclaredService(HAL_INSTANCE);
        if (binder == null) {
            Log.e(TAG, "vendor HAL not declared/available: " + HAL_INSTANCE);
            return null;
        }
        return IMiniVAiHal.Stub.asInterface(binder);
    }

    /** Re-fetch mHal if we don't currently have a live reference. */
    private IMiniVAiHal hal() {
        IMiniVAiHal hal = mHal;
        if (hal == null) {
            hal = fetchHal();
            mHal = hal;
        }
        return hal;
    }

    @Override
    public boolean isReady() {
        IMiniVAiHal hal = hal();
        if (hal == null) {
            return false;
        }
        try {
            return hal.isReady();
        } catch (RemoteException e) {
            Log.e(TAG, "isReady() failed", e);
            mHal = null; // force re-fetch next call
            return false;
        }
    }

    @Override
    public boolean createSession(int sessionId) {
        IMiniVAiHal hal = hal();
        if (hal == null) {
            return false;
        }
        try {
            int ret = hal.createSession(sessionId);
            if (ret != 0) {
                Log.w(TAG, "createSession(" + sessionId + ") failed, HAL returned " + ret);
            }
            return ret == 0;
        } catch (RemoteException e) {
            Log.e(TAG, "createSession(" + sessionId + ") failed", e);
            mHal = null;
            return false;
        }
    }

    @Override
    public boolean destroySession(int sessionId) {
        mSeenSessions.remove(sessionId);

        IMiniVAiHal hal = hal();
        if (hal == null) {
            return false;
        }
        try {
            int ret = hal.destroySession(sessionId);
            return ret == 0;
        } catch (RemoteException e) {
            Log.e(TAG, "destroySession(" + sessionId + ") failed", e);
            mHal = null;
            return false;
        }
    }

    @Override
    public void infer(int sessionId, String prompt, int maxTokens, TokenCallback callback) {
        IMiniVAiHal hal = hal();
        if (hal == null) {
            callback.onError(ErrorCode.GENERIC_FAILURE, "vendor HAL unavailable");
            return;
        }

        IMiniVAiStreamCallback.Stub halCallback = new IMiniVAiStreamCallback.Stub() {
            @Override
            public void onToken(int sid, String token) {
                callback.onToken(token);
            }

            @Override
            public void onComplete(int sid) {
                callback.onComplete();
            }

            @Override
            public void onError(int sid, int code, String message) {
                callback.onError(mapInternalError(code), message);
            }

            @Override
            public String getInterfaceHash() {
                return null;
            }

            @Override
            public int getInterfaceVersion() {
                return IMiniVAiStreamCallback.VERSION;
            }
        };

        try {
            String formattedPrompt = applyChatTemplate(sessionId, prompt);
            int clampedMaxTokens = clampMaxTokens(formattedPrompt, maxTokens);
            if (clampedMaxTokens < maxTokens) {
                Log.w(TAG, "maxTokens clamped from " + maxTokens + " to " + clampedMaxTokens
                        + " to fit context window (nCtx=" + MODEL_N_CTX + ")");
            }
            int ret = hal.inferStream(sessionId, formattedPrompt, clampedMaxTokens, halCallback);
            if (ret != 0) {
                Log.w(TAG, "inferStream(" + sessionId + ") rejected, HAL returned " + ret);
                callback.onError(ErrorCode.GENERIC_FAILURE,
                        "vendor HAL rejected inferStream (code " + ret + ")");
            }
        } catch (RemoteException e) {
            Log.e(TAG, "inferStream(" + sessionId + ") failed", e);
            mHal = null;
            callback.onError(ErrorCode.GENERIC_FAILURE, "vendor HAL call failed: " + e.getMessage());
        }
    }

    @Override
    public void inferSingle(int sessionId, String prompt, int maxTokens, TokenCallback callback) {
        try {
            android.os.SystemProperties.set("persist.vendor.miniv.decode_mode", "single");
        } catch (Exception e) {
            Log.w(TAG, "Failed to set persist.vendor.miniv.decode_mode property", e);
        }
        infer(sessionId, prompt, maxTokens, callback);
    }

    @Override
    public void inferMulti(int sessionId, String prompt, int maxTokens, TokenCallback callback) {
        try {
            android.os.SystemProperties.set("persist.vendor.miniv.decode_mode", "multi");
        } catch (Exception e) {
            Log.w(TAG, "Failed to set persist.vendor.miniv.decode_mode property", e);
        }
        infer(sessionId, prompt, maxTokens, callback);
    }

    @Override
    public void cancel(int sessionId) {
        IMiniVAiHal hal = hal();
        if (hal == null) {
            return;
        }
        try {
            hal.cancel(sessionId);
        } catch (RemoteException e) {
            Log.e(TAG, "cancel(" + sessionId + ") failed", e);
            mHal = null;
        }
    }

    @Override
    public String getModelInfo() {
        IMiniVAiHal hal = hal();
        if (hal == null) {
            return TAG + " (vendor HAL unavailable)";
        }
        try {
            return hal.getModelInfo();
        } catch (RemoteException e) {
            Log.e(TAG, "getModelInfo() failed", e);
            mHal = null;
            return TAG + " (call failed)";
        }
    }

    /**
     * Translate a vendor-HAL-internal error code (IMiniVAiStreamCallback's
     * -50x namespace) into the common LLMEngine.ErrorCode taxonomy (-100x).
     * Deliberately 1:1 by design between the two AIDL interfaces.
     */
    @Override
    public int mapInternalError(int internalCode) {
        switch (internalCode) {
            case InternalErrorCode.SESSION_NOT_FOUND:
                return ErrorCode.SESSION_EVICTED;
            case InternalErrorCode.CACHE_LIMIT_EXCEEDED:
                return ErrorCode.CACHE_LIMIT_EXCEEDED;
            case InternalErrorCode.GENERIC_FAILURE:
            default:
                return ErrorCode.GENERIC_FAILURE;
        }
    }
}