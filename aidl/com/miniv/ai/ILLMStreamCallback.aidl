package com.miniv.ai;

/**
 * MINI-V AI Stream Callback AIDL Interface
 */
oneway interface ILLMStreamCallback {
    // Error codes reported via onError()
    const int ERROR_SESSION_EVICTED = -301;
    const int ERROR_CACHE_LIMIT_EXCEEDED = -302;
    const int ERROR_GENERIC_FAILURE = -303;

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