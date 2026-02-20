import { useCallback, useRef } from 'react'

const useBlurEffect = () => {
  const tempCanvasRef = useRef(null)
  
  // Initialize temp canvas
  const getTempCanvas = useCallback((width, height) => {
    if (!tempCanvasRef.current) {
      tempCanvasRef.current = document.createElement('canvas')
    }
    tempCanvasRef.current.width = width
    tempCanvasRef.current.height = height
    return tempCanvasRef.current
  }, [])

  // Apply blur effect with focus on tracked object
  const applyBlurEffect = useCallback((ctx, video, trackedObject, blurIntensity) => {
    const { width, height } = ctx.canvas
    
    if (!trackedObject || blurIntensity === 0) {
      // No tracking or blur - just draw video
      ctx.drawImage(video, 0, 0, width, height)
      return
    }
    
    const [x, y, w, h] = trackedObject.bbox
    
    // Get temp canvas for operations
    const tempCanvas = getTempCanvas(width, height)
    const tempCtx = tempCanvas.getContext('2d')
    
    // Step 1: Draw blurred background
    ctx.filter = `blur(${blurIntensity}px)`
    ctx.drawImage(video, 0, 0, width, height)
    ctx.filter = 'none'
    
    // Step 2: Create mask for sharp region (tracked object)
    tempCtx.clearRect(0, 0, width, height)
    tempCtx.drawImage(video, 0, 0, width, height)
    
    // Step 3: Draw sharp region with feathered edges
    ctx.save()
    
    // Create elliptical clip region with padding
    const padding = 20
    const centerX = x + w / 2
    const centerY = y + h / 2
    const radiusX = w / 2 + padding
    const radiusY = h / 2 + padding
    
    ctx.beginPath()
    ctx.ellipse(centerX, centerY, radiusX, radiusY, 0, 0, Math.PI * 2)
    ctx.clip()
    
    // Draw sharp region
    ctx.drawImage(video, 0, 0, width, height)
    
    ctx.restore()
    
    // Add subtle vignette effect around focus area
    const gradient = ctx.createRadialGradient(
      centerX, centerY, Math.max(radiusX, radiusY),
      centerX, centerY, Math.max(radiusX, radiusY) * 2
    )
    gradient.addColorStop(0, 'rgba(0, 0, 0, 0)')
    gradient.addColorStop(1, 'rgba(0, 0, 0, 0.2)')
    
    ctx.fillStyle = gradient
    ctx.fillRect(0, 0, width, height)
    
  }, [getTempCanvas])

  // Apply segmentation-based blur (for future MediaPipe integration)
  const applySegmentationBlur = useCallback((ctx, video, segmentation, blurIntensity) => {
    // This will be implemented with MediaPipe selfie segmentation
    // For now, falls back to simple blur
    const { width, height } = ctx.canvas
    ctx.filter = `blur(${blurIntensity}px)`
    ctx.drawImage(video, 0, 0, width, height)
    ctx.filter = 'none'
  }, [])

  return {
    applyBlurEffect,
    applySegmentationBlur
  }
}

export default useBlurEffect
