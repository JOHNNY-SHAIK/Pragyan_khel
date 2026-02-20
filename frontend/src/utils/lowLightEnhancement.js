// Low-light image enhancement utilities

export const enhanceLowLight = (imageData, intensity = 1.5) => {
  const data = imageData.data
  
  for (let i = 0; i < data.length; i += 4) {
    // Get RGB values
    let r = data[i]
    let g = data[i + 1]
    let b = data[i + 2]
    
    // Calculate luminance
    const luminance = (0.299 * r + 0.587 * g + 0.114 * b) / 255
    
    // Apply adaptive enhancement based on luminance
    const enhanceFactor = intensity * (1 - luminance * 0.5)
    
    // Enhance darker pixels more
    data[i] = Math.min(255, r * enhanceFactor)
    data[i + 1] = Math.min(255, g * enhanceFactor)
    data[i + 2] = Math.min(255, b * enhanceFactor)
  }
  
  return imageData
}

export const adjustGamma = (imageData, gamma = 1.2) => {
  const data = imageData.data
  const gammaCorrection = 1 / gamma
  
  // Create lookup table for performance
  const lut = new Uint8Array(256)
  for (let i = 0; i < 256; i++) {
    lut[i] = Math.min(255, Math.pow(i / 255, gammaCorrection) * 255)
  }
  
  for (let i = 0; i < data.length; i += 4) {
    data[i] = lut[data[i]]
    data[i + 1] = lut[data[i + 1]]
    data[i + 2] = lut[data[i + 2]]
  }
  
  return imageData
}

export const autoEnhance = (ctx, video, options = {}) => {
  const { width, height } = ctx.canvas
  
  // Draw video to canvas
  ctx.drawImage(video, 0, 0, width, height)
  
  // Get image data
  let imageData = ctx.getImageData(0, 0, width, height)
  
  // Calculate average brightness
  let totalBrightness = 0
  for (let i = 0; i < imageData.data.length; i += 4) {
    totalBrightness += (imageData.data[i] + imageData.data[i + 1] + imageData.data[i + 2]) / 3
  }
  const avgBrightness = totalBrightness / (width * height)
  
  // Apply enhancement if scene is dark
  if (avgBrightness < 80) {
    const intensity = 1 + (80 - avgBrightness) / 80
    imageData = enhanceLowLight(imageData, Math.min(intensity, 2))
    
    if (avgBrightness < 50) {
      imageData = adjustGamma(imageData, 1.3)
    }
    
    ctx.putImageData(imageData, 0, 0)
  }
}

export default {
  enhanceLowLight,
  adjustGamma,
  autoEnhance
}
