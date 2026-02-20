import React from 'react'
import { motion } from 'framer-motion'

const ModeCard = ({ icon: Icon, title, description, isActive, onClick }) => {
  return (
    <motion.div
      whileHover={{ scale: 1.02 }}
      whileTap={{ scale: 0.98 }}
      onClick={onClick}
      className={`p-6 rounded-2xl cursor-pointer transition-all duration-300 ${
        isActive 
          ? 'bg-gradient-to-br from-primary/20 to-accent/20 border border-accent/50 glow-accent' 
          : 'glass-card hover:border-primary/30'
      }`}
    >
      <div className={`w-12 h-12 rounded-xl flex items-center justify-center mb-4 ${
        isActive ? 'bg-accent/20' : 'bg-dark-600'
      }`}>
        <Icon className={`w-6 h-6 ${isActive ? 'text-accent' : 'text-gray-400'}`} />
      </div>
      
      <h3 className={`text-lg font-semibold mb-2 ${
        isActive ? 'text-accent' : 'text-white'
      }`}>
        {title}
      </h3>
      
      <p className="text-sm text-gray-400">
        {description}
      </p>
      
      {isActive && (
        <div className="mt-4 flex items-center gap-2 text-sm text-accent">
          <div className="w-2 h-2 rounded-full bg-accent animate-pulse" />
          Active
        </div>
      )}
    </motion.div>
  )
}

export default ModeCard
