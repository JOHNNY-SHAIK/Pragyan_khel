// Simple 2D Kalman Filter for smooth object tracking

export class KalmanFilter {
  constructor(initialState) {
    // State: [x, y, vx, vy]
    this.state = {
      x: initialState.x,
      y: initialState.y,
      vx: 0,
      vy: 0
    }
    
    // Process noise
    this.processNoise = 0.1
    
    // Measurement noise
    this.measurementNoise = 2
    
    // Error covariance
    this.errorCovariance = 1
  }
  
  predict() {
    // Simple prediction: position += velocity
    this.state.x += this.state.vx
    this.state.y += this.state.vy
    
    // Increase uncertainty
    this.errorCovariance += this.processNoise
    
    return {
      x: this.state.x,
      y: this.state.y
    }
  }
  
  update(measuredX, measuredY) {
    // Calculate Kalman gain
    const kalmanGain = this.errorCovariance / 
      (this.errorCovariance + this.measurementNoise)
    
    // Update velocity estimation
    this.state.vx = (measuredX - this.state.x) * 0.5
    this.state.vy = (measuredY - this.state.y) * 0.5
    
    // Update position with measurement
    this.state.x += kalmanGain * (measuredX - this.state.x)
    this.state.y += kalmanGain * (measuredY - this.state.y)
    
    // Update error covariance
    this.errorCovariance = (1 - kalmanGain) * this.errorCovariance
    
    return {
      x: this.state.x,
      y: this.state.y
    }
  }
  
  getState() {
    return { ...this.state }
  }
  
  reset(newState) {
    this.state = {
      x: newState.x,
      y: newState.y,
      vx: 0,
      vy: 0
    }
    this.errorCovariance = 1
  }
}

export default KalmanFilter
