package com.miniv.ai;

/**
 * MINI-V AI Stream Callback AIDL Interface
 */
oneway interface ILLMStreamCallback {
    /**
     * Callback method when token created
     */
    void onToken(int sessionId, String token);

    /**
     * Callback method when inference completed
     */
    void onComplete(int sessionId);

    /**
     * Callback method when inference error
     */
    void onError(int sessionId, int code, String message);
}