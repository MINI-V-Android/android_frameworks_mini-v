package com.miniv.ai;

import com.miniv.ai.ILLMStreamCallback;

/**
 * MINI-V AI Service AIDL Interface
 */
interface IMINIVAIService {
    /**
     * Check if AI Service is ready
     */
    boolean isReady();

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
    int inferStream(String prompt, int maxTokens, ILLMStreamCallback callback);

    /**
     * Cancel inference
     * 
     * @param sessionId Session ID to cancel
     */
    void cancel(int sessionId);

    /**
     * Get model info String
     */
    String getModelInfo();
}