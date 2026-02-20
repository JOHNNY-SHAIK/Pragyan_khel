import React from 'react'
import { Link } from 'react-router-dom'
import { motion } from 'framer-motion'
import { 
  Play, 
  Sparkles, 
  Zap, 
  Shield, 
  Eye, 
  Cpu, 
  Camera,
  ArrowRight,
  CheckCircle
} from 'lucide-react'

const Home = () => {
  const features = [
    {
      icon: Eye,
      title: 'Tap-to-Focus',
      description: 'Click any object to instantly lock focus and blur the background'
    },
    {
      icon: Zap,
      title: 'Real-Time Tracking',
      description: 'AI follows your subject seamlessly as it moves through the frame'
    },
    {
      icon: Sparkles,
      title: 'Dynamic Blur',
      description: 'Professional depth-of-field effect that adapts to scene changes'
    },
    {
      icon: Shield,
      title: '100% Private',
      description: 'All processing happens locally in your browser - no data leaves your device'
    },
    {
      icon: Cpu,
      title: 'AI-Powered',
      description: 'Advanced neural networks detect and classify objects instantly'
    },
    {
      icon: Camera,
      title: 'Low-Light Boost',
      description: 'Intelligent enhancement for better tracking in dark environments'
    }
  ]

  const stats = [
    { value: '30+', label: 'FPS Performance' },
    { value: '80+', label: 'Object Classes' },
    { value: '100%', label: 'Browser-Based' },
    { value: '0ms', label: 'Cloud Latency' }
  ]

  return (
    <div className="pt-20">
      {/* Hero Section */}
      <section className="min-h-screen flex items-center justify-center px-6 py-20 relative overflow-hidden">
        {/* Background effects */}
        <div className="absolute inset-0 overflow-hidden">
          <div className="absolute top-1/4 left-1/4 w-96 h-96 bg-primary/20 rounded-full blur-3xl" />
          <div className="absolute bottom-1/4 right-1/4 w-96 h-96 bg-accent/20 rounded-full blur-3xl" />
        </div>
        
        <div className="max-w-7xl mx-auto text-center relative z-10">
          <motion.div
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8 }}
          >
            {/* Badge */}
            <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full glass mb-8">
              <Sparkles className="w-4 h-4 text-accent" />
              <span className="text-sm text-gray-300">AI-Powered Focus Technology</span>
            </div>
            
            {/* Main Heading */}
            <h1 className="text-5xl md:text-7xl font-bold mb-6 leading-tight">
              <span className="text-white">Transform Your Camera Into</span>
              <br />
              <span className="gradient-text">Intelligent Cinema</span>
            </h1>
            
            {/* Subtitle */}
            <p className="text-xl text-gray-400 max-w-3xl mx-auto mb-10">
              Select any object. AI tracks it. Background blurs beautifully. 
              All in real-time, right in your browser. No downloads, no cloud, pure magic.
            </p>
            
            {/* CTA Buttons */}
            <div className="flex flex-col sm:flex-row items-center justify-center gap-4">
              <Link to="/demo" className="btn-primary flex items-center gap-2 text-lg px-8 py-4">
                <Play className="w-5 h-5" />
                Try Live Demo
              </Link>
              <Link to="/features" className="btn-secondary flex items-center gap-2 text-lg px-8 py-4">
                Learn More
                <ArrowRight className="w-5 h-5" />
              </Link>
            </div>
          </motion.div>
          
          {/* Demo Preview */}
          <motion.div
            initial={{ opacity: 0, y: 50 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, delay: 0.3 }}
            className="mt-16"
          >
            <div className="relative max-w-4xl mx-auto">
              <div className="aspect-video rounded-2xl bg-dark-800 border border-dark-600 overflow-hidden glow-accent">
                <div className="absolute inset-0 flex items-center justify-center">
                  <div className="text-center">
                    <div className="w-20 h-20 rounded-full bg-gradient-to-r from-primary to-accent flex items-center justify-center mx-auto mb-4 animate-pulse">
                      <Camera className="w-10 h-10 text-white" />
                    </div>
                    <p className="text-gray-400">Live demo preview</p>
                  </div>
                </div>
              </div>
              
              {/* Floating elements */}
              <div className="absolute -right-4 top-1/4 glass-card px-4 py-3 animate-float">
                <p className="text-sm text-accent font-semibold">Object Detected</p>
                <p className="text-xs text-gray-400">Person • 98%</p>
              </div>
              
              <div className="absolute -left-4 bottom-1/4 glass-card px-4 py-3 animate-float" style={{ animationDelay: '1s' }}>
                <p className="text-sm text-green-400 font-semibold">Tracking Active</p>
                <p className="text-xs text-gray-400">30 FPS • Stable</p>
              </div>
            </div>
          </motion.div>
        </div>
      </section>
      
      {/* Stats Section */}
      <section className="py-20 border-y border-dark-600">
        <div className="max-w-7xl mx-auto px-6">
          <div className="grid grid-cols-2 md:grid-cols-4 gap-8">
            {stats.map((stat, index) => (
              <motion.div
                key={stat.label}
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                transition={{ delay: index * 0.1 }}
                className="text-center"
              >
                <div className="text-4xl md:text-5xl font-bold gradient-text mb-2">
                  {stat.value}
                </div>
                <div className="text-gray-400">{stat.label}</div>
              </motion.div>
            ))}
          </div>
        </div>
      </section>
      
      {/* Features Section */}
      <section className="py-24 px-6">
        <div className="max-w-7xl mx-auto">
          <motion.div
            initial={{ opacity: 0 }}
            whileInView={{ opacity: 1 }}
            className="text-center mb-16"
          >
            <h2 className="text-4xl font-bold text-white mb-4">
              Powerful Features
            </h2>
            <p className="text-xl text-gray-400 max-w-2xl mx-auto">
              Everything you need to create professional-looking content with AI focus
            </p>
          </motion.div>
          
          <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
            {features.map((feature, index) => (
              <motion.div
                key={feature.title}
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                transition={{ delay: index * 0.1 }}
                className="glass-card p-8 hover:border-accent/30 transition-colors group"
              >
                <div className="w-14 h-14 rounded-xl bg-gradient-to-br from-primary/20 to-accent/20 flex items-center justify-center mb-6 group-hover:scale-110 transition-transform">
                  <feature.icon className="w-7 h-7 text-accent" />
                </div>
                <h3 className="text-xl font-semibold text-white mb-3">
                  {feature.title}
                </h3>
                <p className="text-gray-400">
                  {feature.description}
                </p>
              </motion.div>
            ))}
          </div>
        </div>
      </section>
      
      {/* How It Works Section */}
      <section className="py-24 px-6 bg-dark-800/50">
        <div className="max-w-7xl mx-auto">
          <motion.div
            initial={{ opacity: 0 }}
            whileInView={{ opacity: 1 }}
            className="text-center mb-16"
          >
            <h2 className="text-4xl font-bold text-white mb-4">
              How It Works
            </h2>
            <p className="text-xl text-gray-400">
              Three simple steps to cinematic focus
            </p>
          </motion.div>
          
          <div className="grid md:grid-cols-3 gap-8">
            {[
              { step: '01', title: 'Start Camera', desc: 'Enable your webcam with one click' },
              { step: '02', title: 'Select Object', desc: 'Tap on any detected object to focus' },
              { step: '03', title: 'Enjoy the Magic', desc: 'AI tracks and blurs automatically' }
            ].map((item, index) => (
              <motion.div
                key={item.step}
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                transition={{ delay: index * 0.2 }}
                className="text-center"
              >
                <div className="text-6xl font-bold text-dark-600 mb-4">{item.step}</div>
                <h3 className="text-2xl font-semibold text-white mb-2">{item.title}</h3>
                <p className="text-gray-400">{item.desc}</p>
              </motion.div>
            ))}
          </div>
        </div>
      </section>
      
      {/* CTA Section */}
      <section className="py-24 px-6">
        <div className="max-w-4xl mx-auto text-center">
          <motion.div
            initial={{ opacity: 0, scale: 0.95 }}
            whileInView={{ opacity: 1, scale: 1 }}
            className="glass-card p-12 md:p-16 glow-primary"
          >
            <h2 className="text-4xl md:text-5xl font-bold text-white mb-6">
              Ready to Transform Your Camera?
            </h2>
            <p className="text-xl text-gray-400 mb-8">
              Experience AI-powered focus technology in seconds. No signup required.
            </p>
            <Link to="/demo" className="btn-primary inline-flex items-center gap-2 text-lg px-10 py-5">
              <Play className="w-5 h-5" />
              Launch Demo Now
            </Link>
            
            <div className="flex items-center justify-center gap-6 mt-8 text-sm text-gray-400">
              <div className="flex items-center gap-2">
                <CheckCircle className="w-4 h-4 text-green-400" />
                Free to use
              </div>
              <div className="flex items-center gap-2">
                <CheckCircle className="w-4 h-4 text-green-400" />
                No signup
              </div>
              <div className="flex items-center gap-2">
                <CheckCircle className="w-4 h-4 text-green-400" />
                100% private
              </div>
            </div>
          </motion.div>
        </div>
      </section>
    </div>
  )
}

export default Home
