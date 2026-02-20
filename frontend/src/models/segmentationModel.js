// MediaPipe Selfie Segmentation Model

let segmenter = null

export const loadSegmentationModel = async () => {
  try {
    // Dynamic import for MediaPipe
    const { SelfieSegmentation } = await import('@mediapipe/selfie_segmentation')
    
    segmenter = new SelfieSegmentation({
      locateFile: (file) => {
        return `https://cdn.jsdelivr.net/npm/@mediapipe/selfie_segmentation/${file}`
      }
    })
    
    segmenter.setOptions({
      modelSelection: 1 // 0 = General, 1 = Landscape
    })
    
    await segmenter.initialize()
    console.log('Segmentation model loaded')
    
    return segmenter
  } catch (error) {
    console.error('Error loading segmentation model:', error)
    return null
  }
}

export const runSegmentation = async (videoElement, onResults) => {
  if (!segmenter) {
    console.warn('Segmentation model not loaded')
    return
  }
  
  segmenter.onResults(onResults)
  await segmenter.send({ image: videoElement })
}

export const applySegmentationMask = (ctx, results, blurAmount = 15) => {
  const { width, height } = ctx.canvas
  
  if (!results.segmentationMask) return
  
  // Save context
  ctx.save()
  
  // Draw blurred background
  ctx.filter = `blur(${blurAmount}px)`
  ctx.drawImage(results.image, 0, 0, width, height)
  ctx.filter = 'none'
  
  // Apply mask
  ctx.globalCompositeOperation = 'destination-in'
  ctx.drawImage(results.segmentationMask, 0, 0, width, height)
  
  // Draw sharp foreground
  ctx.globalCompositeOperation = 'destination-over'
  ctx.drawImage(results.image, 0, 0, width, height)
  
  // Restore context
  ctx.restore()
}

export default {
  loadSegmentationModel,
  runSegmentation,
  applySegmentationMask
}
