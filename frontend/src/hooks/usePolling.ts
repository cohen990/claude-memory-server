import { useEffect, useRef } from 'react'

/**
 * Calls `callback` immediately, then every `interval` ms.
 * Cleans up on unmount or when deps change.
 * Pass `enabled: false` to pause polling.
 */
export function usePolling(
  callback: () => void,
  interval: number,
  enabled = true,
) {
  const savedCallback = useRef(callback)
  savedCallback.current = callback

  useEffect(() => {
    if (!enabled) return
    savedCallback.current()
    const id = setInterval(() => savedCallback.current(), interval)
    return () => clearInterval(id)
  }, [interval, enabled])
}
