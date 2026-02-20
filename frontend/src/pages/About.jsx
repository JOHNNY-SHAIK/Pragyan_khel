import React from 'react'
import { motion } from 'framer-motion'
import { Link } from 'react-router-dom'
import { 
  Target, 
  Heart, 
  Rocket, 
  Users, 
  Code, 
  Lightbulb,
  Trophy,
  ArrowRight
} from 'lucide-react'

const About = () => {
  const values = [
    {
      icon: Target,
      title: 'Mission',
      description: 'To democratize professional video effects by making AI-powered focus technology accessible to everyone, directly in the browser.'
    },
    {
      icon: Heart,
      title: 'Privacy First',
      description: 'We believe your data belongs to you. That\'s why all processing happens locally - your video never touches our servers.'
    },
    {
      icon: Rocket,
      title: 'Innovation',
      description: 'Pushing the boundaries of what\'s possible with browser-based AI, delivering desktop-quality effects on any device.'
    }
  ]

  const timeline = [
    { date: 'Concept', title: 'Idea Born', desc: 'Identified the need for accessible AI focus tools' },
    { date: 'Research', title: 'Tech Exploration', desc: 'Evaluated TensorFlow.js, MediaPipe, and WebGL' },
    { date: 'Development', title: 'Building', desc: 'Created core detection and tracking system' },
    { date: 'Launch', title: 'Hackathon', desc: 'Presenting FocusAI to the world' }
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
            About FocusAI
          </h1>
          <p className="text-xl text-gray-400 max-w-3xl mx-auto">
            Transforming how the world captures and shares moments through 
            intelligent, accessible AI technology
          </p>
        </motion.div>

        {/* Hero Quote */}
        <motion.div
          initial={{ opacity: 0, scale: 0.95 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ delay: 0.2 }}
          className="glass-card p-12 text-center mb-20 glow-primary"
        >
          <blockquote className="text-2xl md:text-3xl font-light text-white italic mb-6">
            "Every camera deserves the power of AI-driven focus. 
            We're making that a reality."
          </blockquote>
          <p className="text-accent font-semibold">— The FocusAI Team</p>
        </motion.div>

        {/* Values */}
        <div className="grid md:grid-cols-3 gap-8 mb-20">
          {values.map((value, index) => (
            <motion.div
              key={value.title}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.3 + index * 0.1 }}
              className="glass-card p-8"
            >
              <div className="w-14 h-14 rounded-xl bg-gradient-to-br from-primary to-accent flex items-center justify-center mb-6">
                <value.icon className="w-7 h-7 text-white" />
              </div>
              <h3 className="text-xl font-semibold text-white mb-4">
                {value.title}
              </h3>
              <p className="text-gray-400">
                {value.description}
              </p>
            </motion.div>
          ))}
        </div>

        {/* Timeline */}
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.5 }}
          className="mb-20"
        >
          <h2 className="text-3xl font-bold text-white text-center mb-12">
            Our Journey
          </h2>
          <div className="relative">
            <div className="absolute left-1/2 transform -translate-x-1/2 h-full w-px bg-dark-600" />
            <div className="space-y-12">
              {timeline.map((item, index) => (
                <motion.div
                  key={item.date}
                  initial={{ opacity: 0, x: index % 2 === 0 ? -50 : 50 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: 0.6 + index * 0.1 }}
                  className={`flex items-center gap-8 ${index % 2 === 0 ? 'flex-row' : 'flex-row-reverse'}`}
                >
                  <div className={`flex-1 ${index % 2 === 0 ? 'text-right' : 'text-left'}`}>
                    <div className="glass-card p-6 inline-block">
                      <p className="text-accent font-semibold mb-1">{item.date}</p>
                      <h4 className="text-lg font-semibold text-white mb-2">{item.title}</h4>
                      <p className="text-gray-400">{item.desc}</p>
                    </div>
                  </div>
                  <div className="w-4 h-4 rounded-full bg-accent flex-shrink-0 relative z-10" />
                  <div className="flex-1" />
                </motion.div>
              ))}
            </div>
          </div>
        </motion.div>

        {/* Hackathon Badge */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.8 }}
          className="glass-card p-8 text-center"
        >
          <Trophy className="w-16 h-16 text-yellow-400 mx-auto mb-6" />
          <h2 className="text-3xl font-bold text-white mb-4">
            Built for Innovation
          </h2>
          <p className="text-xl text-gray-400 max-w-2xl mx-auto mb-8">
            FocusAI was created to showcase the power of browser-based AI 
            and push the boundaries of what's possible with modern web technologies.
          </p>
          <Link to="/demo" className="btn-primary inline-flex items-center gap-2">
            Experience It Now
            <ArrowRight className="w-5 h-5" />
          </Link>
        </motion.div>
      </div>
    </div>
  )
}

export default About
