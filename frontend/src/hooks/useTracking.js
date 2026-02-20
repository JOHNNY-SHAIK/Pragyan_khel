import { useState, useCallback, useRef } from 'react'
import { KalmanFilter } from '../utils/kalmanFilter'

const useTracking = () => {
  const [trackedObject, setTrackedObject] = useState(null)
  const [trackingStats, setTrackingStats] = useState({
    fps: 0,
    stability: 0,
    speed: 0
  })
  
  const kalmanFilterRef = useRef(null)
  const lastFrameTime = useRef(Date.now())
  const frameCount = useRef(0)
  const lastPosition = useRef(null)

  // Initialize Kalman filter
  const initKalmanFilter = useCallback((bbox) => {
    const [x, y, w, h] = bbox
    kalmanFilterRef.current = new KalmanFilter({
      x: x + w / 2,
      y: y + h / 2
    })
  }, [])

  // Calculate IOU between two bounding boxes
  const calculateIOU = useCallback((box1, box2) => {
    const [x1, y1, w1, h1] = box1
    const [x2, y2, w2, h2] = box2
    
    const xA = Math.max(x1, x2)
    const yA = Math.max(y1, y2)
    const xB = Math.min(x1 + w1, x2 + w2)
    const yB = Math.min(y1 + h1, y2 + h2)
    
    const intersectionArea = Math.max(0, xB - xA) * Math.max(0, yB - yA)
    const box1Area = w1 * h1
    const box2Area = w2 * h2
    
    return intersectionArea / (box1Area + box2Area - intersectionArea)
  }, [])

  // Update tracking with new detections
  const updateTracking = useCallback((detections) => {
    if (!trackedObject || !detections.length) return
    
    // Find best matching detection using IOU
    let bestMatch = null
    let bestIOU = 0.3 // Minimum IOU threshold
    
    for (const detection of detections) {
      // Also check if same class
      if (detection.class !== trackedObject.class) continue
      
      const iou = calculateIOU(trackedObject.bbox, detection.bbox)
      if (iou > bestIOU) {
        bestIOU = iou
        bestMatch = detection
      }
    }
    
    if (bestMatch) {
      // Apply Kalman filter smoothing
      if (!kalmanFilterRef.current) {
        initKalmanFilter(bestMatch.bbox)
      }
      
      const [x, y, w, h] = bestMatch.bbox
      const smoothed = kalmanFilterRef.current.update(x + w / 2, y + h / 2)
      
      // Calculate speed
      let speed = 0
      if (lastPosition.current) {
        const dx = smoothed.x - lastPosition.current.x
        const dy = smoothed.y - lastPosition.current.y
        speed = Math.sqrt(dx * dx + dy * dy)
      }
      lastPosition.current = { x: smoothed.x, y: smoothed.y }
      
      // Update tracked object with smoothed position
      setTrackedObject(prev => ({
        ...prev,
        ...bestMatch,
        bbox: [
          smoothed.x - w / 2,
          smoothed.y - h / 2,
          w,
          h
        ]
      }))
      
      // Update FPS
      frameCount.current++
      const now = Date.now()
      if (now - lastFrameTime.current >= 1000) {
        setTrackingStats(prev => ({
          ...prev,
          fps: frameCount.current,
          speed: Math.round(speed),
          stability: Math.round(bestIOU * 100)
        }))
        frameCount.current = 0
        lastFrameTime.current = now
      }
    }
  }, [trackedObject, calculateIOU, initKalmanFilter])

  // Clear tracking
  const clearTracking = useCallback(() => {
    setTrackedObject(null)
    kalmanFilterRef.current = null
    lastPosition.current = null
  }, [])

  return {
    trackedObject,
    setTrackedObject,
    updateTracking,
    clearTracking,
    trackingStats
  }
}

export default useTracking
