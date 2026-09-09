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
    // — the common taxonomy every LLMEngine implementation must translate its own
    // internal errors into (see SimpleLLMEngine.InternalErrorCode for an
    // example of a concrete engine's private codes)
    interface ErrorCode {
        // Session is already evicted
        int SESSION_EVICTED = -1001;
        // Session cache limit has exceed
        int CACHE_LIMIT_EXCEEDED = -1002;
        // Engine-specific failure
        int GENERIC_FAILURE = -1003;
    }

    /**
     * Translate an engine-implementation-internal error code into the
     * common ErrorCode taxonomy above. Every implementation must provide
     * this — it is the one place internal codes get converted before ever
     * reaching TokenCallback.onError(). The internal codes themselves stay
     * private to each implementation; only this translation is a contract.
     *
     * @param internalCode Implementation-specific internal error code
     */
    int mapInternalError(int internalCode);

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
     * Run inference and stream tokens via callback (Default / Auto mode)
     *
     * @param sessionId Session ID for this inference
     * @param prompt    Input prompt
     * @param maxTokens Maximum number of tokens to generate
     * @param callback  Token stream callback
     */
    void infer(int sessionId, String prompt, int maxTokens, TokenCallback callback);

    /**
     * Run Single-Decode inference (N=1, Real-time immediate streaming)
     */
    default void inferSingle(int sessionId, String prompt, int maxTokens, TokenCallback callback) {
        infer(sessionId, prompt, maxTokens, callback);
    }

    /**
     * Run Multi-Decode inference (N=2, Best-of-N candidate selection)
     */
    default void inferMulti(int sessionId, String prompt, int maxTokens, TokenCallback callback) {
        infer(sessionId, prompt, maxTokens, callback);
    }

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