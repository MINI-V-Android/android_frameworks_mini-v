package com.miniv.ai.service;

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
    // Executor
    private final ExecutorService mExecutor;
    // Session ID
    private final AtomicInteger mNextSessionId = new AtomicInteger(1);

    public MINIVAIService() {
        mEngine = new SimpleLLMEngine();
        mExecutor = Executors.newCachedThreadPool();
        Log.i(TAG, "Initialized MINIVAIService");
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
     * Request inference to Model
     * 
     * @param prompt Input prompt to model
     * @param maxTokens Maximum number of tokens to generate
     * @param callback Callback to receive inference results
     * 
     * @return Session ID
     *      Negative if failed to create session
     */
    @Override
    public int inferStream(String prompt, int maxTokens, ILLMStreamCallback callback) {
        // Check if engine is ready
        if (!mEngine.isReady()) {
            Log.w(TAG, "inferStream() called but engine is not ready");
            return -1;
        }

        // Check if parameter available
        if (prompt == null || prompt.isEmpty() || maxTokens <= 0 || callback == null) {
            Log.w(TAG, "inferStream() called with invalid parameters");
            return -2;
        }

        // Current session id
        final int sessionId = mNextSessionId.getAndIncrement();

        // Run engine inference with Executor
        mExecutor.submit(() -> {
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
                    try {
                        // Callback onError
                        callback.onError(sessionId, code, message);
                    } catch (RemoteException e) {
                        Log.w(TAG, "onError failed, session " + sessionId, e);
                    }
                }
            });
        });

        // Return current session id
        return sessionId;
    }

    /**
     * Cancel inference
     * 
     * @param sessionId Session ID to cancel
     */
    @Override
    public void cancel(int sessionId) {
        Log.i(TAG, "cancel() requested for session " + sessionId);
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
}