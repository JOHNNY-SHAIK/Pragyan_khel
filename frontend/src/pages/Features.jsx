import React from 'react'
import { motion } from 'framer-motion'
import { 
  Eye, 
  Zap, 
  Lock, 
  Cpu, 
  Camera, 
  Sparkles,
  BarChart3,
  Gauge,
  Moon,
  Layers,
  RefreshCw,
  Smartphone
} from 'lucide-react'

const Features = () => {
  const mainFeatures = [
    {
      icon: Eye,
      title: 'Tap-to-Focus Selection',
      description: 'Simply click on any object detected by our AI to instantly lock focus. The selected subject stays sharp while everything else beautifully blurs away.',
      color: 'from-blue-500 to-cyan-500'
    },
    {
      icon: Zap,
      title: 'Real-Time Object Tracking',
      description: 'Our advanced tracking algorithm follows your subject as it moves through the frame. Smooth, stable tracking at up to 30 FPS with Kalman filter smoothing.',
      color: 'from-yellow-500 to-orange-500'
    },
    {
      icon: Sparkles,
      title: 'Dynamic Background Blur',
      description: 'Professional bokeh-style blur that adapts in real-time. Adjustable intensity from subtle separation to dramatic depth-of-field effects.',
      color: 'from-purple-500 to-pink-500'
    },
    {
      icon: Lock,
      title: '100% Privacy-First',
      description: 'All AI processing happens locally in your browser using WebGL acceleration. Your video never leaves your device - zero cloud, zero compromise.',
      color: 'from-green-500 to-emerald-500'
    }
  ]

  const additionalFeatures = [
    { icon: Cpu, title: 'Neural Network Detection', description: 'COCO-SSD model detects 80+ object classes' },
    { icon: BarChart3, title: 'Live Analytics', description: 'Real-time FPS, confidence, and stability metrics' },
    { icon: Gauge, title: 'Performance Optimized', description: 'WebGL acceleration for smooth 30+ FPS' },
    { icon: Moon, title: 'Low-Light Enhancement', description: 'Adaptive brightness for dark environments' },
    { icon: Layers, title: 'Multiple Modes', description: 'Specialized modes for people, objects, pets' },
    { icon: RefreshCw, title: 'Instant Switching', description: 'Change focus targets seamlessly' },
    { icon: Camera, title: 'Any Camera', description: 'Works with webcam or external cameras' },
    { icon: Smartphone, title: 'Mobile Ready', description: 'Responsive design for all devices' }
  ]

  const techStack = [
    { name: 'TensorFlow.js', desc: 'Neural network inference' },
    { name: 'COCO-SSD', desc: 'Object detection model' },
    { name: 'MediaPipe', desc: 'Segmentation (optional)' },
    { name: 'React', desc: 'UI framework' },
    { name: 'WebGL', desc: 'GPU acceleration' },
    { name: 'Canvas API', desc: 'Real-time rendering' }
  ]

  return (
    <div className="pt-24 pb-20 px-6">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="text-center mb-20"
        >
          <h1 className="text-5xl font-bold text-white mb-6">
            Powerful Features
          </h1>
          <p className="text-xl text-gray-400 max-w-3xl mx-auto">
            Built with cutting-edge AI technology to deliver professional-grade 
            focus effects directly in your browser
          </p>
        </motion.div>

        {/* Main Features */}
        <div className="grid md:grid-cols-2 gap-8 mb-20">
          {mainFeatures.map((feature, index) => (
            <motion.div
              key={feature.title}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: index * 0.1 }}
              className="glass-card p-8 group hover:border-accent/30 transition-all"
            >
              <div className={`w-16 h-16 rounded-2xl bg-gradient-to-br ${feature.color} flex items-center justify-center mb-6 group-hover:scale-110 transition-transform`}>
                <feature.icon className="w-8 h-8 text-white" />
              </div>
              <h3 className="text-2xl font-semibold text-white mb-4">
                {feature.title}
              </h3>
              <p className="text-gray-400 leading-relaxed">
                {feature.description}
              </p>
            </motion.div>
          ))}
        </div>

        {/* Additional Features Grid */}
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.4 }}
          className="mb-20"
        >
          <h2 className="text-3xl font-bold text-white text-center mb-12">
            And Much More
          </h2>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            {additionalFeatures.map((feature, index) => (
              <motion.div
                key={feature.title}
                initial={{ opacity: 0, scale: 0.9 }}
                animate={{ opacity: 1, scale: 1 }}
                transition={{ delay: 0.5 + index * 0.05 }}
                className="glass-card p-6 text-center hover:border-accent/30 transition-colors"
              >
                <feature.icon className="w-8 h-8 text-accent mx-auto mb-3" />
                <h4 className="font-semibold text-white mb-2">{feature.title}</h4>
                <p className="text-sm text-gray-400">{feature.description}</p>
              </motion.div>
            ))}
          </div>
        </motion.div>

        {/* Tech Stack */}
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.6 }}
          className="glass-card p-8"
        >
          <h2 className="text-3xl font-bold text-white text-center mb-8">
            Built With Modern Tech
          </h2>
          <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-4">
            {techStack.map((tech) => (
              <div
                key={tech.name}
                className="p-4 rounded-xl bg-dark-700/50 text-center hover:bg-dark-600/50 transition-colors"
              >
                <p className="font-semibold text-accent mb-1">{tech.name}</p>
                <p className="text-xs text-gray-400">{tech.desc}</p>
              </div>
            ))}
          </div>
        </motion.div>
      </div>
    </div>
  )
}

export default Features
