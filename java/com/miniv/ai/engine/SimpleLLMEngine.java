package com.miniv.ai.engine;

import java.util.Map;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * Simulation LLM Engine Implementation
 * - Simulates simple LLM inference processing
 * - Just returns input prompt into words with random delay
 */
public class SimpleLLMEngine implements LLMEngine {
    // Constants
    private static final String TAG = "SimpleLLMEngine";
    private static final int TOKEN_DELAY_MS = 100;

    // Active session IDs
    private final Set<Integer> mSessions = ConcurrentHashMap.newKeySet();
    // <sessionId, cancelled> flags
    private final Map<Integer, AtomicBoolean> mCancelFlags = new ConcurrentHashMap<>();

    /**
     * Check if engine is ready or not
     */
    @Override
    public boolean isReady() {
        // Simple engine is always ready
        return true;
    }

    /**
     * Create a new session
     *
     * @param sessionId Session ID to create, assigned by the caller
     *
     * @return true if the session was created successfully
     */
    @Override
    public boolean createSession(int sessionId) {
        // Simple engine has no real resource to allocate, just track the id
        return mSessions.add(sessionId);
    }

    /**
     * Destroy a session and release its resources
     *
     * @param sessionId Session ID to destroy
     *
     * @return true if the session existed and was destroyed
     */
    @Override
    public boolean destroySession(int sessionId) {
        // Clear any leftover cancel flag
        mCancelFlags.remove(sessionId);

        // Drop tracked session
        return mSessions.remove(sessionId);
    }

    /**
     * Run inference and stream tokens via callback
     *
     * @param sessionId Session ID for this inference
     * @param prompt    Input prompt
     * @param maxTokens Maximum number of tokens to generate
     * @param callback  Token stream callback
     */
    @Override
    public void infer(int sessionId, String prompt, int maxTokens, TokenCallback callback) {
        // Check if session exists
        if (!mSessions.contains(sessionId)) {
            callback.onError(ErrorCode.SESSION_EVICTED, "unknown session: " + sessionId);
            return;
        }

        // Set cancelled flag as false
        AtomicBoolean cancelled = new AtomicBoolean(false);
        mCancelFlags.put(sessionId, cancelled);

        try {
            // Split prompt with "Space"
            String[] words = prompt.split(" ");
            int tokenCount = 0;

            for (String word : words) {
                // If already cancelled session, break
                if (cancelled.get()) {
                    break;
                }

                // If token exceed, break
                if (tokenCount >= maxTokens) {
                    break;
                }

                // Send onToken callback with splitted word
                callback.onToken(word + " ");
                tokenCount++;

                // Sleep with delay
                Thread.sleep(TOKEN_DELAY_MS);
            }

            // Send onComplete callback if not cancelled
            if (!cancelled.get()) {
                callback.onComplete();
            }
        } catch (InterruptedException e) {
            // If any exception caught, kill and send onError callback
            Thread.currentThread().interrupt();
            callback.onError(-1, "Interrupted");
        } finally {
            // Clear cancelled flag info
            mCancelFlags.remove(sessionId);
        }
    }

    /**
     * Cancel ongoing inference for given session
     *
     * @param sessionId Session ID to cancel
     */
    @Override
    public void cancel(int sessionId) {
        // Set cancelled flag as true
        AtomicBoolean flag = mCancelFlags.get(sessionId);
        if (flag != null) {
            flag.set(true);
        }
    }

    /**
     * Get model info string
     */
    @Override
    public String getModelInfo() {
        return TAG;
    }
}