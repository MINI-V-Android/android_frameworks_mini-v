package com.miniv.ai.engine;

/**
 * LLM Engine Interface
 */
public interface LLMEngine {
    // Token Callback
    interface TokenCallback {
        void onToken(String token);

        void onComplete();

        void onError(int code, String message);
    }

    // Error codes reported via TokenCallback.onError()
    interface ErrorCode {
        // Session is already evicted
        int SESSION_EVICTED = -10;
        // Session cache limit has exceed
        int CACHE_LIMIT_EXCEEDED = -11;
    }

    /**
     * Check if engine is ready or not
     */
    boolean isReady();

    /**
     * Create a new session
     *
     * @param sessionId Session ID to create, assigned by the caller
     *
     * @return true if the session was created successfully
     */
    boolean createSession(int sessionId);

    /**
     * Destroy a session and release its resources
     *
     * @param sessionId Session ID to destroy
     *
     * @return true if the session existed and was destroyed
     */
    boolean destroySession(int sessionId);

    /**
     * Run inference and stream tokens via callback
     *
     * @param sessionId Session ID for this inference
     * @param prompt    Input prompt
     * @param maxTokens Maximum number of tokens to generate
     * @param callback  Token stream callback
     */
    void infer(int sessionId, String prompt, int maxTokens, TokenCallback callback);

    /**
     * Cancel ongoing inference for given session
     *
     * @param sessionId Session ID to cancel
     */
    void cancel(int sessionId);

    /**
     * Get model info string
     */
    String getModelInfo();
}