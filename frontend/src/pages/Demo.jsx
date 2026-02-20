import React, { useState } from 'react'
import { motion } from 'framer-motion'
import { User, Target, Zap, Eye, EyeOff, Sun, Moon, RefreshCw, Sliders } from 'lucide-react'
import VideoCanvas from '../components/VideoCanvas'

const Demo = () => {
  const [blurIntensity, setBlurIntensity] = useState(15)
  const [autoFocus, setAutoFocus] = useState(false)
  const [lowLightMode, setLowLightMode] = useState(false)
  const [showDetections, setShowDetections] = useState(true)

  return (
    <div className="pt-24 pb-12 px-6 min-h-screen">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="text-center mb-8"
        >
          <h1 className="text-4xl font-bold text-white mb-3">
            ?? FocusAI - Cricket Edition
          </h1>
          <p className="text-lg text-gray-400 max-w-3xl mx-auto">
            Upload a cricket video or use your webcam. YOLOv8 detects players, ball, and bat.
            Click any object to focus — OpenCV applies cinematic background blur in real-time.
          </p>
        </motion.div>

        {/* Tech Stack Badges */}
        <div className="flex flex-wrap justify-center gap-3 mb-8">
          {['YOLOv8', 'OpenCV', 'Deep SORT', 'MediaPipe', 'Kalman Filter', 'Flask'].map(tech => (
            <span key={tech} className="px-3 py-1 rounded-full glass text-sm text-accent font-medium">
              {tech}
            </span>
          ))}
        </div>

        {/* Main Layout */}
        <div className="grid lg:grid-cols-[1fr_300px] gap-6">
          {/* Video Canvas */}
          <motion.div
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ delay: 0.2 }}
          >
            <VideoCanvas />
          </motion.div>

          {/* Side Panel */}
          <motion.div
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: 0.3 }}
            className="glass-card p-6"
          >
            <h3 className="text-lg font-semibold text-white mb-6 flex items-center gap-2">
              <Sliders className="w-5 h-5 text-accent" />
              Controls
            </h3>
            
            <div className="space-y-5">
              {/* Auto Focus */}
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <Target className="w-4 h-4 text-gray-400" />
                  <span className="text-sm text-gray-300">Auto Focus (Ball Priority)</span>
                </div>
                <button
                  onClick={() => setAutoFocus(!autoFocus)}
                  className={`w-11 h-6 rounded-full transition-colors ${autoFocus ? 'bg-accent' : 'bg-dark-600'}`}
                >
                  <div className={`w-5 h-5 rounded-full bg-white transform transition-transform ${autoFocus ? 'translate-x-5' : 'translate-x-0.5'}`} />
                </button>
              </div>
              
              {/* Show Detections */}
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  {showDetections ? <Eye className="w-4 h-4 text-gray-400" /> : <EyeOff className="w-4 h-4 text-gray-400" />}
                  <span className="text-sm text-gray-300">Show Detections</span>
                </div>
                <button
                  onClick={() => setShowDetections(!showDetections)}
                  className={`w-11 h-6 rounded-full transition-colors ${showDetections ? 'bg-accent' : 'bg-dark-600'}`}
                >
                  <div className={`w-5 h-5 rounded-full bg-white transform transition-transform ${showDetections ? 'translate-x-5' : 'translate-x-0.5'}`} />
                </button>
              </div>
              
              {/* Low Light */}
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  {lowLightMode ? <Moon className="w-4 h-4 text-gray-400" /> : <Sun className="w-4 h-4 text-gray-400" />}
                  <span className="text-sm text-gray-300">Low Light Boost</span>
                </div>
                <button
                  onClick={() => setLowLightMode(!lowLightMode)}
                  className={`w-11 h-6 rounded-full transition-colors ${lowLightMode ? 'bg-accent' : 'bg-dark-600'}`}
                >
                  <div className={`w-5 h-5 rounded-full bg-white transform transition-transform ${lowLightMode ? 'translate-x-5' : 'translate-x-0.5'}`} />
                </button>
              </div>
              
              {/* Divider */}
              <div className="border-t border-dark-600 pt-4">
                <h4 className="text-sm font-semibold text-gray-400 mb-3">Detection Classes</h4>
                <div className="space-y-2">
                  {[
                    { name: 'Player', color: '#00FF00', emoji: '??' },
                    { name: 'Ball', color: '#FF0000', emoji: '??' },
                    { name: 'Bat', color: '#FFA500', emoji: '??' },
                  ].map(cls => (
                    <div key={cls.name} className="flex items-center gap-2 text-sm">
                      <div className="w-3 h-3 rounded-full" style={{ backgroundColor: cls.color }} />
                      <span className="text-gray-300">{cls.emoji} {cls.name}</span>
                    </div>
                  ))}
                </div>
              </div>
              
              {/* Pipeline Info */}
              <div className="border-t border-dark-600 pt-4">
                <h4 className="text-sm font-semibold text-gray-400 mb-3">AI Pipeline</h4>
                <div className="space-y-2 text-xs text-gray-500">
                  <p>1. ?? Video Input</p>
                  <p>2. ?? YOLOv8 Detection</p>
                  <p>3. ?? Deep SORT Tracking</p>
                  <p>4. ?? Kalman Filter Smoothing</p>
                  <p>5. ?? Focus Selection</p>
                  <p>6. ??? OpenCV Bokeh Blur</p>
                  <p>7. ?? Output Render</p>
                </div>
              </div>
            </div>
          </motion.div>
        </div>

        {/* Instructions */}
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.5 }}
          className="mt-8 glass-card p-6"
        >
          <h3 className="text-lg font-semibold text-white mb-4">?? How To Use</h3>
          <div className="grid md:grid-cols-4 gap-6">
            {[
              { step: '1', title: 'Upload Video', desc: 'Upload the cricket video provided' },
              { step: '2', title: 'AI Detects', desc: 'YOLOv8 detects players, ball, bat' },
              { step: '3', title: 'Click to Focus', desc: 'Click any object to track it' },
              { step: '4', title: 'Magic Blur', desc: 'Background blurs cinematically' }
            ].map(item => (
              <div key={item.step} className="flex items-start gap-3">
                <div className="w-8 h-8 rounded-full bg-accent/20 flex items-center justify-center flex-shrink-0">
                  <span className="text-accent font-bold">{item.step}</span>
                </div>
                <div>
                  <p className="font-medium text-white">{item.title}</p>
                  <p className="text-sm text-gray-400">{item.desc}</p>
                </div>
              </div>
            ))}
          </div>
        </motion.div>
      </div>
    </div>
  )
}

export default Demo
