package com.miniv.ai.service;

import android.os.IBinder;
import android.os.RemoteException;
import android.util.Log;

import com.miniv.ai.ILLMStreamCallback;
import com.miniv.ai.IMINIVAIService;
import com.miniv.ai.engine.LLMEngine;
import com.miniv.ai.engine.SimpleLLMEngine;

import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * MINI-V AI Service
 *
 * A system service that bridges the app layer and [LLMEngine] layer.
 */
public class MINIVAIService extends IMINIVAIService.Stub {
    // Constants
    private static final String TAG = "MINIVAIService";

    // LLM Engine instance
    private final LLMEngine mEngine;
    // Session ID
    private final AtomicInteger mNextSessionId = new AtomicInteger(1);
    // Active session info
    // Key: Session ID
    // Value: Record instance
    private final ConcurrentHashMap<Integer, SessionRecord> mSessions = new ConcurrentHashMap<>();

    public MINIVAIService() {
        mEngine = new SimpleLLMEngine();
        Log.i(TAG, "Initialized MINIVAIService");
    }

    /**
     * Record of each session
     * 
     * - Created in createSession(), removed in destroySession()
     */
    private static class SessionRecord {
        // Executor that serializes all requests for this session
        final ExecutorService executor = Executors.newSingleThreadExecutor();
        // Queued/in-flight task, used by cancel()
        volatile Future<?> currentTask;
        // Death recipient linked to the callback passed into inferStream()
        volatile IBinder.DeathRecipient deathRecipient;
        // Binder the death recipient above is linked to
        volatile IBinder linkedCallbackBinder;
    }

    /**
     * Check if AI Service is ready
     *
     * @return Is engine ready, or not
     */
    @Override
    public boolean isReady() {
        return mEngine.isReady();
    }

    /**
     * Create a new conversation session.
     *
     * @return Session ID
     *         Negative if failed to create session
     *         Error codes is defined at [IMINIVAIService]
     */
    @Override
    public int createSession() {
        // Check if engine is ready
        if (!mEngine.isReady()) {
            Log.w(TAG, "createSession() called but engine is not ready");
            return CREATE_SESSION_ERR_ENGINE_NOT_READY;
        }

        // Current session id
        final int sessionId = mNextSessionId.getAndIncrement();

        // Allocate session on engine
        if (!mEngine.createSession(sessionId)) {
            Log.e(TAG, "engine.createSession() failed for session " + sessionId);
            return CREATE_SESSION_ERR_ALLOC_FAILED;
        }

        mSessions.put(sessionId, new SessionRecord());
        Log.i(TAG, "createSession: " + sessionId);

        // Return current session id
        return sessionId;
    }

    /**
     * Destroy a session and release its resources (cancels any in-flight
     * work first).
     *
     * @param sessionId Session ID to destroy
     *
     * @return true if the session existed and was destroyed
     */
    @Override
    public boolean destroySession(int sessionId) {
        // Check if session exists
        SessionRecord record = mSessions.remove(sessionId);
        if (record == null) {
            Log.w(TAG, "destroySession: unknown session " + sessionId);
            return false;
        }

        // Stop watching for client death
        unlinkDeath(record);

        // Cancel queued/in-flight work
        if (record.currentTask != null) {
            record.currentTask.cancel(true);
        }
        mEngine.cancel(sessionId);

        // Shut down the session's Executor
        record.executor.shutdownNow();
        try {
            record.executor.awaitTermination(5, TimeUnit.SECONDS);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        }

        // Release session on engine
        boolean destroyed = mEngine.destroySession(sessionId);
        Log.i(TAG, "destroySession: " + sessionId + " (engineDestroyed=" + destroyed + ")");
        return destroyed;
    }

    /**
     * Queue an inference request onto the given session. Returns immediately
     * after the request is queued — does not wait for completion. If another
     * request on the same session is still running, this one simply starts
     * after it finishes (appears as "slower", not as an error).
     *
     * @param sessionId Session ID to run inference on
     * @param prompt    Input prompt to model
     * @param maxTokens Maximum number of tokens to generate
     * @param callback  Callback to receive inference results
     *
     * @return 0 if queued successfully
     *         Negative otherwise
     *         Error codes is defined at [IMINIVAIService]
     */
    @Override
    public int inferStream(int sessionId, String prompt, int maxTokens, ILLMStreamCallback callback) {
        // Check if engine is ready
        if (!mEngine.isReady()) {
            Log.w(TAG, "inferStream() called but engine is not ready");
            return INFER_ERR_ENGINE_NOT_READY;
        }

        // Check if parameter available
        if (prompt == null || prompt.isEmpty() || maxTokens <= 0 || callback == null) {
            Log.w(TAG, "inferStream() called with invalid parameters");
            return INFER_ERR_INVALID_PARAMS;
        }

        // Check if session exists
        SessionRecord record = mSessions.get(sessionId);
        if (record == null) {
            Log.w(TAG, "inferStream: unknown session " + sessionId);
            return INFER_ERR_UNKNOWN_SESSION;
        }

        // Add watching for client death
        if (!linkDeath(sessionId, record, callback)) {
            return INFER_ERR_LINK_DEATH_FAILED;
        }

        // Queue engine inference on the session's Executor
        record.currentTask = record.executor.submit(() -> {
            mEngine.infer(sessionId, prompt, maxTokens, new LLMEngine.TokenCallback() {
                @Override
                public void onToken(String token) {
                    try {
                        // Callback onToken
                        callback.onToken(sessionId, token);
                    } catch (RemoteException e) {
                        // Callback cancel
                        Log.w(TAG, "onToken failed, cancelling session " + sessionId, e);
                        mEngine.cancel(sessionId);
                    }
                }

                @Override
                public void onComplete() {
                    try {
                        // Callback onComplete
                        callback.onComplete(sessionId);
                    } catch (RemoteException e) {
                        Log.w(TAG, "onComplete failed, session " + sessionId, e);
                    }
                }

                @Override
                public void onError(int code, String message) {
                    // Translate the engine-internal error code into the
                    // AIDL-level contract — callers should never need to
                    // know about LLMEngine's internals
                    int reportedCode = translateErrorCode(code);

                    try {
                        // Callback onError
                        callback.onError(sessionId, reportedCode, message);
                    } catch (RemoteException e) {
                        Log.w(TAG, "onError failed, session " + sessionId, e);
                    }

                    // Vendor already discarded this session
                    // -> destroy local session info
                    if (code == LLMEngine.ErrorCode.SESSION_EVICTED) {
                        destroySession(sessionId);
                    }
                }
            });
        });

        // Return success
        return 0;
    }

    /**
     * Cancel in-flight or queued work for a session.
     *
     * @param sessionId Session ID to cancel
     */
    @Override
    public void cancel(int sessionId) {
        Log.i(TAG, "cancel() requested for session " + sessionId);

        // Cancel queued/in-flight task, if any
        SessionRecord record = mSessions.get(sessionId);
        if (record != null && record.currentTask != null) {
            record.currentTask.cancel(true);
        }
        mEngine.cancel(sessionId);
    }

    /**
     * Get model info String
     *
     * @return model info from engine
     */
    @Override
    public String getModelInfo() {
        return mEngine.getModelInfo();
    }

    /**
     * Translate an LLMEngine.ErrorCode into the ILLMStreamCallback contract.
     * The app should never need to know about LLMEngine's internal codes.
     *
     * @param engineCode One of LLMEngine.ErrorCode's values
     */
    private static int translateErrorCode(int engineCode) {
        switch (engineCode) {
            case LLMEngine.ErrorCode.SESSION_EVICTED:
                return ILLMStreamCallback.ERROR_SESSION_EVICTED;
            case LLMEngine.ErrorCode.CACHE_LIMIT_EXCEEDED:
                return ILLMStreamCallback.ERROR_CACHE_LIMIT_EXCEEDED;
            default:
                return ILLMStreamCallback.ERROR_GENERIC_FAILURE;
        }
    }

    /**
     * Link death detection to the callback binder passed into inferStream().
     *
     * @param sessionId Session ID this callback belongs to
     * @param record    Session record to attach the death recipient to
     * @param callback  Callback whose binder should be watched
     *
     * @return true if linked (or already linked to this callback)
     */
    private boolean linkDeath(int sessionId, SessionRecord record, ILLMStreamCallback callback) {
        IBinder binder = callback.asBinder();
        if (record.linkedCallbackBinder == binder) {
            // Already linked to this exact callback
            return true;
        }

        // Drop any stale link from a previous callback instance
        unlinkDeath(record);

        IBinder.DeathRecipient recipient = () -> {
            // Callback cancel
            Log.w(TAG, "Client died, destroying session " + sessionId);
            destroySession(sessionId);
        };
        try {
            binder.linkToDeath(recipient, 0);
        } catch (RemoteException e) {
            Log.e(TAG, "Failed to linkToDeath for session " + sessionId, e);
            return false;
        }

        record.deathRecipient = recipient;
        record.linkedCallbackBinder = binder;
        return true;
    }

    /**
     * Unlink a previously linked death recipient, if any.
     *
     * @param record Session record to detach the death recipient from
     */
    private void unlinkDeath(SessionRecord record) {
        if (record.linkedCallbackBinder != null && record.deathRecipient != null) {
            record.linkedCallbackBinder.unlinkToDeath(record.deathRecipient, 0);
        }
        record.deathRecipient = null;
        record.linkedCallbackBinder = null;
    }
}