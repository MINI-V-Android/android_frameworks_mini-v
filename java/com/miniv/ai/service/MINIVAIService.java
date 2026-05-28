package com.miniv.ai.service;

/**
 * MINI-V AI Service
 *
 * A system service that bridges the app layer and [LLMEngine] layer.
 */
public class MINIVAIService extends IMINIVAIService.Stub {
    // Constants
    private static final String TAG = "MINIVAIService";

    public MINIVAIService() {
        // TODO: Init service instance
    }

    /**
     * Check if AI Service is ready
     */
    @Override
    public boolean isReady() {
        // TODO: Implement
        return false;
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
        // TODO: Implement
        return 0;
    }

    /**
     * Cancel inference
     * 
     * @param sessionId Session ID to cancel
     */
    @Override
    public void cancel(int sessionId) {
        // TODO: Implement
    }

    /**
     * Get model info String
     */
    @Override
    public String getModelInfo() {
        // TODO: Implement
        return TAG;
    }
}