package com.miniv.ai;

import com.miniv.ai.ILLMStreamCallback;

/**
 * MINI-V AI Service AIDL Interface
 */
interface IMINIVAIService {
    // Error codes for createSession()
    const int CREATE_SESSION_ERR_ENGINE_NOT_READY = -101;
    const int CREATE_SESSION_ERR_ALLOC_FAILED = -102;
 
    // Error codes for inferStream()
    const int INFER_ERR_ENGINE_NOT_READY = -201;
    const int INFER_ERR_INVALID_PARAMS = -202;
    const int INFER_ERR_LINK_DEATH_FAILED = -203;
    const int INFER_ERR_UNKNOWN_SESSION = -204;

    /**
     * Check if AI Service is ready
     */
    boolean isReady();

    /**
     * Create a new conversation session.
     *
     * @return Session ID
     *      Negative if failed to create session
     */
    int createSession();
 
    /**
     * Destroy a session and release its resources (cancels any in-flight
     * work first).
     *
     * @return true if the session existed and was destroyed
     */
    boolean destroySession(int sessionId);

    /**
     * Queue an inference request onto the given session. Returns immediately
     * after the request is queued — does not wait for completion. If another
     * request on the same session is still running, this one simply starts
     * after it finishes (appears as "slower", not as an error).
     * 
     * @param sessionId Session ID to run inference on
     * @param prompt Input prompt to model
     * @param maxTokens Maximum number of tokens to generate
     * @param callback Callback to receive inference results
     * 
     * @return 0 if queued successfully
     *      Negative otherwise
     */
    int inferStream(int sessionId, String prompt, int maxTokens, ILLMStreamCallback callback);

    /**
     * Cancel in-flight or queued work for a session.
     * 
     * @param sessionId Session ID to cancel
     */
    void cancel(int sessionId);

    /**
     * Get model info String
     */
    String getModelInfo();
}