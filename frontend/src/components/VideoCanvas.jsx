import React, { useRef, useEffect, useState, useCallback } from 'react'
import { motion } from 'framer-motion'
import { Camera, CameraOff, Upload, Maximize2, Settings, Play, Pause, RotateCcw } from 'lucide-react'
import AnalyticsOverlay from './AnalyticsOverlay'

const API_URL = 'http://localhost:5000'

const VideoCanvas = () => {
  const videoRef = useRef(null)
  const canvasRef = useRef(null)
  const fileInputRef = useRef(null)
  const streamImgRef = useRef(null)
  
  const [mode, setMode] = useState('none') // none, webcam, video, stream
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState(null)
  const [videoInfo, setVideoInfo] = useState(null)
  const [detections, setDetections] = useState([])
  const [trackedObjects, setTrackedObjects] = useState([])
  const [focusTarget, setFocusTarget] = useState(null)
  const [blurIntensity, setBlurIntensity] = useState(15)
  const [isProcessing, setIsProcessing] = useState(false)
  const [processProgress, setProcessProgress] = useState(0)
  const [backendStatus, setBackendStatus] = useState('checking')
  const [isPlaying, setIsPlaying] = useState(false)

  // Check backend health
  useEffect(() => {
    const checkBackend = async () => {
      try {
        const res = await fetch(`${API_URL}/api/health`)
        const data = await res.json()
        setBackendStatus(data.status === 'running' ? 'connected' : 'error')
      } catch {
        setBackendStatus('disconnected')
      }
    }
    checkBackend()
    const interval = setInterval(checkBackend, 5000)
    return () => clearInterval(interval)
  }, [])

  // Upload video
  const handleVideoUpload = async (e) => {
    const file = e.target.files[0]
    if (!file) return
    
    setIsLoading(true)
    setError(null)
    
    try {
      // Show video preview
      const url = URL.createObjectURL(file)
      if (videoRef.current) {
        videoRef.current.src = url
        videoRef.current.load()
      }
      
      // Upload to backend
      const formData = new FormData()
      formData.append('video', file)
      
      const res = await fetch(`${API_URL}/api/upload`, {
        method: 'POST',
        body: formData
      })
      
      const data = await res.json()
      
      if (data.success) {
        setVideoInfo(data.video_info)
        setMode('video')
      } else {
        setError(data.error)
      }
    } catch (err) {
      setError('Failed to upload video. Is the backend running?')
      console.error(err)
    } finally {
      setIsLoading(false)
    }
  }

  // Start webcam
  const startWebcam = async () => {
    setIsLoading(true)
    setError(null)
    
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: 1280, height: 720 },
        audio: false
      })
      
      if (videoRef.current) {
        videoRef.current.srcObject = stream
        await videoRef.current.play()
        setMode('webcam')
        setIsPlaying(true)
        startWebcamDetection()
      }
    } catch (err) {
      setError('Camera access denied')
    } finally {
      setIsLoading(false)
    }
  }

  // Send webcam frames to backend for detection
  const startWebcamDetection = () => {
    const canvas = document.createElement('canvas')
    
    const sendFrame = async () => {
      if (!videoRef.current || mode !== 'webcam') return
      
      canvas.width = videoRef.current.videoWidth
      canvas.height = videoRef.current.videoHeight
      const ctx = canvas.getContext('2d')
      ctx.drawImage(videoRef.current, 0, 0)
      
      const frameData = canvas.toDataURL('image/jpeg', 0.7)
      
      try {
        const res = await fetch(`${API_URL}/api/detect_frame`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ frame: frameData })
        })
        
        const data = await res.json()
        setDetections(data.detections || [])
        setTrackedObjects(data.tracked_objects || [])
        
        // Draw detections on canvas
        drawDetections(data.detections, data.tracked_objects)
      } catch (err) {
        console.error('Detection error:', err)
      }
      
      if (mode === 'webcam') {
        requestAnimationFrame(sendFrame)
      }
    }
    
    sendFrame()
  }

  // Draw detections on overlay canvas
  const drawDetections = (dets, tracked) => {
    const canvas = canvasRef.current
    if (!canvas || !videoRef.current) return
    
    const ctx = canvas.getContext('2d')
    canvas.width = videoRef.current.videoWidth || videoRef.current.clientWidth
    canvas.height = videoRef.current.videoHeight || videoRef.current.clientHeight
    
    ctx.clearRect(0, 0, canvas.width, canvas.height)
    
    const scaleX = canvas.width / (videoInfo?.width || canvas.width)
    const scaleY = canvas.height / (videoInfo?.height || canvas.height)
    
    dets?.forEach(det => {
      const [x, y, w, h] = det.bbox
      const isTracked = focusTarget?.track_id !== undefined && 
        tracked?.some(t => t.track_id === focusTarget.track_id && t.class === det.class)
      
      // Box color based on class
      const colors = {
        'player': '#00FF00', 'ball': '#FF0000', 'bat': '#FFA500',
        'person': '#00FF00', 'sports ball': '#FF0000'
      }
      const color = isTracked ? '#00E5FF' : (colors[det.class] || '#FFFFFF')
      
      ctx.strokeStyle = color
      ctx.lineWidth = isTracked ? 3 : 2
      ctx.strokeRect(x, y, w, h)
      
      // Label
      ctx.fillStyle = color
      ctx.font = 'bold 14px Inter'
      ctx.fillText(`${det.class} ${Math.round(det.confidence * 100)}%`, x, y - 5)
      
      // Corner markers for tracked
      if (isTracked) {
        ctx.strokeStyle = '#FF5722'
        ctx.lineWidth = 4
        const cs = 15
        ctx.beginPath(); ctx.moveTo(x, y + cs); ctx.lineTo(x, y); ctx.lineTo(x + cs, y); ctx.stroke()
        ctx.beginPath(); ctx.moveTo(x + w - cs, y); ctx.lineTo(x + w, y); ctx.lineTo(x + w, y + cs); ctx.stroke()
        ctx.beginPath(); ctx.moveTo(x, y + h - cs); ctx.lineTo(x, y + h); ctx.lineTo(x + cs, y + h); ctx.stroke()
        ctx.beginPath(); ctx.moveTo(x + w - cs, y + h); ctx.lineTo(x + w, y + h); ctx.lineTo(x + w, y + h - cs); ctx.stroke()
      }
    })
  }

  // Handle click to select focus target
  const handleCanvasClick = (e) => {
    const canvas = canvasRef.current
    const rect = canvas.getBoundingClientRect()
    const clickX = (e.clientX - rect.left) * (canvas.width / rect.width)
    const clickY = (e.clientY - rect.top) * (canvas.height / rect.height)
    
    // Find clicked detection
    for (const det of detections) {
      const [x, y, w, h] = det.bbox
      if (clickX >= x && clickX <= x + w && clickY >= y && clickY <= y + h) {
        const target = {
          track_id: trackedObjects.find(t => t.class === det.class)?.track_id,
          class: det.class,
          bbox: det.bbox
        }
        setFocusTarget(target)
        
        // Send to backend
        fetch(`${API_URL}/api/set_focus`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ 
            target, 
            blur_intensity: blurIntensity 
          })
        }).catch(console.error)
        
        break
      }
    }
  }

  // Process video with focus effect
  const processVideo = async () => {
    if (!videoInfo) return
    
    setIsProcessing(true)
    setProcessProgress(0)
    
    try {
      const res = await fetch(`${API_URL}/api/process_video`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          focus_target: focusTarget,
          blur_intensity: blurIntensity
        })
      })
      
      const data = await res.json()
      
      if (data.success) {
        setProcessProgress(100)
        // Show processed video stream
        setMode('stream')
      }
    } catch (err) {
      setError('Processing failed')
    } finally {
      setIsProcessing(false)
    }
  }

  // Stream processed video
  const startStream = () => {
    if (streamImgRef.current) {
      streamImgRef.current.src = `${API_URL}/api/stream_processed?t=${Date.now()}`
    }
  }

  // Detect objects in current video frame
  const detectCurrentFrame = async () => {
    if (!videoRef.current || mode !== 'video') return
    
    const canvas = document.createElement('canvas')
    canvas.width = videoRef.current.videoWidth
    canvas.height = videoRef.current.videoHeight
    const ctx = canvas.getContext('2d')
    ctx.drawImage(videoRef.current, 0, 0)
    
    const frameData = canvas.toDataURL('image/jpeg', 0.8)
    
    try {
      const res = await fetch(`${API_URL}/api/detect_frame`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ frame: frameData })
      })
      
      const data = await res.json()
      setDetections(data.detections || [])
      setTrackedObjects(data.tracked_objects || [])
      drawDetections(data.detections, data.tracked_objects)
    } catch (err) {
      console.error('Detection error:', err)
    }
  }

  // Video playback with detection
  useEffect(() => {
    if (mode !== 'video' || !isPlaying) return
    
    const interval = setInterval(detectCurrentFrame, 200) // 5 detections per second
    return () => clearInterval(interval)
  }, [mode, isPlaying, focusTarget])

  // Stop everything
  const stop = () => {
    if (videoRef.current?.srcObject) {
      videoRef.current.srcObject.getTracks().forEach(t => t.stop())
      videoRef.current.srcObject = null
    }
    if (videoRef.current) {
      videoRef.current.pause()
      videoRef.current.src = ''
    }
    setMode('none')
    setDetections([])
    setTrackedObjects([])
    setFocusTarget(null)
    setIsPlaying(false)
  }

  return (
    <div className="w-full">
      {/* Backend Status */}
      <div className={`mb-4 flex items-center gap-2 px-3 py-2 rounded-lg text-sm ${
        backendStatus === 'connected' ? 'bg-green-500/10 text-green-400' :
        backendStatus === 'checking' ? 'bg-yellow-500/10 text-yellow-400' :
        'bg-red-500/10 text-red-400'
      }`}>
        <div className={`w-2 h-2 rounded-full ${
          backendStatus === 'connected' ? 'bg-green-400 animate-pulse' :
          backendStatus === 'checking' ? 'bg-yellow-400' : 'bg-red-400'
        }`} />
        {backendStatus === 'connected' ? 'Backend Connected (YOLOv8 + OpenCV)' :
         backendStatus === 'checking' ? 'Checking backend...' :
         'Backend not running - Start with: python app.py'}
      </div>

      <div className="video-container relative bg-dark-800 rounded-2xl overflow-hidden aspect-video">
        {/* Video Element */}
        <video
          ref={videoRef}
          className={`absolute inset-0 w-full h-full object-contain ${mode === 'stream' ? 'hidden' : ''}`}
          playsInline
          muted
          onPlay={() => setIsPlaying(true)}
          onPause={() => setIsPlaying(false)}
          controls={mode === 'video'}
        />
        
        {/* Stream Image */}
        {mode === 'stream' && (
          <img
            ref={streamImgRef}
            className="absolute inset-0 w-full h-full object-contain"
            alt="Processed stream"
          />
        )}
        
        {/* Detection Overlay Canvas */}
        <canvas
          ref={canvasRef}
          className="absolute inset-0 w-full h-full cursor-crosshair"
          onClick={handleCanvasClick}
          style={{ pointerEvents: mode !== 'none' ? 'auto' : 'none' }}
        />
        
        {/* Loading */}
        {isLoading && (
          <div className="absolute inset-0 flex items-center justify-center bg-dark-900/80 z-10">
            <div className="text-center">
              <div className="loading-spinner w-12 h-12 mx-auto mb-4" />
              <p className="text-gray-400">Loading...</p>
            </div>
          </div>
        )}
        
        {/* No Video State */}
        {mode === 'none' && !isLoading && (
          <div className="absolute inset-0 flex items-center justify-center z-10">
            <div className="text-center">
              <div className="w-24 h-24 rounded-full bg-dark-700 flex items-center justify-center mx-auto mb-6">
                <Camera className="w-12 h-12 text-gray-500" />
              </div>
              <h3 className="text-xl font-semibold text-white mb-6">Choose Input Source</h3>
              
              <div className="flex flex-col sm:flex-row gap-4 justify-center">
                <button onClick={startWebcam} className="btn-primary flex items-center gap-2">
                  <Camera size={18} />
                  Start Webcam
                </button>
                
                <button 
                  onClick={() => fileInputRef.current?.click()} 
                  className="btn-secondary flex items-center gap-2"
                >
                  <Upload size={18} />
                  Upload Cricket Video
                </button>
              </div>
              
              <input
                ref={fileInputRef}
                type="file"
                accept="video/*"
                onChange={handleVideoUpload}
                className="hidden"
              />
            </div>
          </div>
        )}
        
        {/* Error */}
        {error && (
          <div className="absolute top-4 left-1/2 -translate-x-1/2 px-4 py-2 bg-red-500/20 text-red-400 rounded-lg text-sm z-20">
            {error}
          </div>
        )}
        
        {/* Focus Target Info */}
        {focusTarget && (
          <motion.div
            initial={{ opacity: 0, y: -20 }}
            animate={{ opacity: 1, y: 0 }}
            className="absolute top-4 right-4 px-4 py-3 rounded-xl glass z-20"
          >
            <p className="text-sm text-gray-400">Focused On:</p>
            <p className="text-lg font-bold text-accent">{focusTarget.class}</p>
            <button 
              onClick={() => setFocusTarget(null)}
              className="text-xs text-gray-500 hover:text-red-400 mt-1"
            >
              Clear Focus
            </button>
          </motion.div>
        )}
        
        {/* Click instruction */}
        {mode !== 'none' && !focusTarget && detections.length > 0 && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            className="absolute top-4 left-1/2 -translate-x-1/2 px-4 py-2 rounded-full glass z-20"
          >
            <p className="text-sm text-accent">?? Click on any detected object to focus</p>
          </motion.div>
        )}
        
        {/* Analytics */}
        {mode !== 'none' && (
          <AnalyticsOverlay
            detections={detections}
            trackedObject={focusTarget}
            trackingStats={{ fps: 30, stability: 95 }}
          />
        )}
        
        {/* Bottom Controls */}
        {mode !== 'none' && (
          <div className="absolute bottom-4 left-1/2 -translate-x-1/2 flex items-center gap-3 z-20">
            {/* Stop */}
            <button onClick={stop} className="p-3 rounded-full bg-red-500/20 text-red-500 hover:bg-red-500/30">
              <CameraOff size={18} />
            </button>
            
            {/* Play/Pause for video */}
            {mode === 'video' && (
              <button 
                onClick={() => {
                  if (videoRef.current?.paused) {
                    videoRef.current.play()
                  } else {
                    videoRef.current?.pause()
                  }
                }}
                className="p-3 rounded-full glass text-white hover:bg-white/10"
              >
                {isPlaying ? <Pause size={18} /> : <Play size={18} />}
              </button>
            )}
            
            {/* Blur slider */}
            <div className="flex items-center gap-2 px-4 py-2 rounded-full glass">
              <span className="text-sm text-gray-400">Blur:</span>
              <input
                type="range"
                min="0"
                max="30"
                value={blurIntensity}
                onChange={(e) => setBlurIntensity(Number(e.target.value))}
                className="w-24 accent-accent"
              />
              <span className="text-sm text-white w-6">{blurIntensity}</span>
            </div>
            
            {/* Process Video Button */}
            {mode === 'video' && focusTarget && (
              <button
                onClick={processVideo}
                disabled={isProcessing}
                className="px-4 py-2 rounded-full bg-accent/20 text-accent hover:bg-accent/30 text-sm font-medium disabled:opacity-50"
              >
                {isProcessing ? `Processing... ${processProgress}%` : '?? Process Video'}
              </button>
            )}
            
            {/* Stream processed */}
            {mode === 'video' && (
              <button
                onClick={() => { setMode('stream'); startStream() }}
                className="px-4 py-2 rounded-full glass text-accent hover:bg-accent/10 text-sm"
              >
                ? Live Stream
              </button>
            )}
          </div>
        )}
      </div>
    </div>
  )
}

export default VideoCanvas
