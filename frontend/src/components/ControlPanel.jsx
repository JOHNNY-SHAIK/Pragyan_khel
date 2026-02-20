import React from 'react'
import { motion } from 'framer-motion'
import { 
  Sliders, 
  Eye, 
  EyeOff, 
  Sun, 
  Moon, 
  Zap, 
  Target,
  RefreshCw 
} from 'lucide-react'

const ControlPanel = ({ 
  blurIntensity, 
  setBlurIntensity,
  autoFocus,
  setAutoFocus,
  lowLightMode,
  setLowLightMode,
  showDetections,
  setShowDetections,
  onReset
}) => {
  return (
    <motion.div
      initial={{ opacity: 0, x: 20 }}
      animate={{ opacity: 1, x: 0 }}
      className="glass-card p-6 w-full lg:w-80"
    >
      <h3 className="text-lg font-semibold text-white mb-6 flex items-center gap-2">
        <Sliders className="w-5 h-5 text-accent" />
        Controls
      </h3>
      
      <div className="space-y-6">
        {/* Blur Intensity */}
        <div>
          <label className="flex items-center justify-between text-sm text-gray-400 mb-2">
            <span>Blur Intensity</span>
            <span className="text-accent">{blurIntensity}px</span>
          </label>
          <input
            type="range"
            min="0"
            max="30"
            value={blurIntensity}
            onChange={(e) => setBlurIntensity(Number(e.target.value))}
            className="w-full h-2 bg-dark-600 rounded-lg appearance-none cursor-pointer accent-accent"
          />
        </div>
        
        {/* Auto Focus Toggle */}
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <Target className="w-5 h-5 text-gray-400" />
            <span className="text-sm text-gray-300">Auto Focus</span>
          </div>
          <button
            onClick={() => setAutoFocus(!autoFocus)}
            className={`w-12 h-6 rounded-full transition-colors ${
              autoFocus ? 'bg-accent' : 'bg-dark-600'
            }`}
          >
            <div className={`w-5 h-5 rounded-full bg-white transform transition-transform ${
              autoFocus ? 'translate-x-6' : 'translate-x-0.5'
            }`} />
          </button>
        </div>
        
        {/* Show Detections Toggle */}
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            {showDetections ? (
              <Eye className="w-5 h-5 text-gray-400" />
            ) : (
              <EyeOff className="w-5 h-5 text-gray-400" />
            )}
            <span className="text-sm text-gray-300">Show Detections</span>
          </div>
          <button
            onClick={() => setShowDetections(!showDetections)}
            className={`w-12 h-6 rounded-full transition-colors ${
              showDetections ? 'bg-accent' : 'bg-dark-600'
            }`}
          >
            <div className={`w-5 h-5 rounded-full bg-white transform transition-transform ${
              showDetections ? 'translate-x-6' : 'translate-x-0.5'
            }`} />
          </button>
        </div>
        
        {/* Low Light Mode Toggle */}
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            {lowLightMode ? (
              <Moon className="w-5 h-5 text-gray-400" />
            ) : (
              <Sun className="w-5 h-5 text-gray-400" />
            )}
            <span className="text-sm text-gray-300">Low Light Boost</span>
          </div>
          <button
            onClick={() => setLowLightMode(!lowLightMode)}
            className={`w-12 h-6 rounded-full transition-colors ${
              lowLightMode ? 'bg-accent' : 'bg-dark-600'
            }`}
          >
            <div className={`w-5 h-5 rounded-full bg-white transform transition-transform ${
              lowLightMode ? 'translate-x-6' : 'translate-x-0.5'
            }`} />
          </button>
        </div>
        
        {/* Performance Mode */}
        <div className="pt-4 border-t border-dark-600">
          <label className="text-sm text-gray-400 mb-3 block">Performance Mode</label>
          <div className="grid grid-cols-3 gap-2">
            {['Low', 'Medium', 'High'].map((mode) => (
              <button
                key={mode}
                className="px-3 py-2 rounded-lg text-sm font-medium transition-colors bg-dark-600 text-gray-400 hover:bg-accent/20 hover:text-accent"
              >
                {mode}
              </button>
            ))}
          </div>
        </div>
        
        {/* Reset Button */}
        <button
          onClick={onReset}
          className="w-full py-3 rounded-xl border border-dark-600 text-gray-400 hover:border-accent hover:text-accent transition-colors flex items-center justify-center gap-2"
        >
          <RefreshCw className="w-4 h-4" />
          Reset All
        </button>
      </div>
    </motion.div>
  )
}

export default ControlPanel
