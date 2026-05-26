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

    /**
     * Check if engine is ready or not
     */
    boolean isReady();

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