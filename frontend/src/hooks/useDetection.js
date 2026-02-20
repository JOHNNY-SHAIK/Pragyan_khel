import { useState, useCallback, useRef } from 'react'
import * as tf from '@tensorflow/tfjs'
import * as cocoSsd from '@tensorflow-models/coco-ssd'

const useDetection = () => {
  const [detections, setDetections] = useState([])
  const [isModelLoaded, setIsModelLoaded] = useState(false)
  const [isLoading, setIsLoading] = useState(false)
  const modelRef = useRef(null)

  // Load COCO-SSD model
  const loadModel = useCallback(async () => {
    if (modelRef.current) return
    
    setIsLoading(true)
    try {
      // Set backend
      await tf.setBackend('webgl')
      await tf.ready()
      
      // Load model
      modelRef.current = await cocoSsd.load({
        base: 'lite_mobilenet_v2' // Lighter model for better performance
      })
      
      setIsModelLoaded(true)
      console.log('COCO-SSD model loaded successfully')
    } catch (error) {
      console.error('Error loading model:', error)
    } finally {
      setIsLoading(false)
    }
  }, [])

  // Run detection on video frame
  const detectObjects = useCallback(async (videoElement) => {
    if (!modelRef.current || !videoElement) return []
    
    try {
      const predictions = await modelRef.current.detect(videoElement)
      
      // Filter by confidence threshold
      const filteredDetections = predictions.filter(p => p.score > 0.5)
      
      setDetections(filteredDetections)
      return filteredDetections
    } catch (error) {
      console.error('Detection error:', error)
      return []
    }
  }, [])

  // Clean up
  const disposeModel = useCallback(() => {
    if (modelRef.current) {
      modelRef.current = null
      setIsModelLoaded(false)
    }
  }, [])

  return {
    detections,
    isModelLoaded,
    isLoading,
    loadModel,
    detectObjects,
    disposeModel
  }
}

export default useDetection
