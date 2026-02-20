import React from 'react'
import { motion } from 'framer-motion'
import { Activity, Clock, Target, Cpu } from 'lucide-react'

const AnalyticsOverlay = ({ detections, trackedObject, trackingStats }) => {
  const fps = trackingStats?.fps || 0
  const trackingTime = trackedObject 
    ? Math.floor((Date.now() - trackedObject.startTime) / 1000) 
    : 0

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      className="absolute top-4 left-4 space-y-2"
    >
      {/* FPS Counter */}
      <div className="flex items-center gap-2 px-3 py-2 rounded-lg glass text-sm">
        <Cpu className="w-4 h-4 text-accent" />
        <span className="text-gray-300">FPS:</span>
        <span className={`font-mono font-bold ${
          fps >= 24 ? 'text-green-400' : fps >= 15 ? 'text-yellow-400' : 'text-red-400'
        }`}>
          {fps}
        </span>
      </div>
      
      {/* Objects Detected */}
      <div className="flex items-center gap-2 px-3 py-2 rounded-lg glass text-sm">
        <Target className="w-4 h-4 text-accent" />
        <span className="text-gray-300">Objects:</span>
        <span className="font-mono font-bold text-white">{detections.length}</span>
      </div>
      
      {/* Tracking Info */}
      {trackedObject && (
        <>
          <div className="flex items-center gap-2 px-3 py-2 rounded-lg glass text-sm">
            <Activity className="w-4 h-4 text-accent" />
            <span className="text-gray-300">Tracking:</span>
            <span className="font-bold text-accent">{trackedObject.class}</span>
          </div>
          
          <div className="flex items-center gap-2 px-3 py-2 rounded-lg glass text-sm">
            <Clock className="w-4 h-4 text-accent" />
            <span className="text-gray-300">Duration:</span>
            <span className="font-mono font-bold text-white">{trackingTime}s</span>
          </div>
          
          {/* Confidence */}
          <div className="px-3 py-2 rounded-lg glass">
            <div className="flex items-center justify-between text-sm mb-1">
              <span className="text-gray-400">Confidence</span>
              <span className="text-accent font-bold">
                {Math.round(trackedObject.score * 100)}%
              </span>
            </div>
            <div className="h-1.5 bg-dark-600 rounded-full overflow-hidden">
              <div 
                className="h-full bg-gradient-to-r from-primary to-accent transition-all"
                style={{ width: `${trackedObject.score * 100}%` }}
              />
            </div>
          </div>
        </>
      )}
    </motion.div>
  )
}

export default AnalyticsOverlay
