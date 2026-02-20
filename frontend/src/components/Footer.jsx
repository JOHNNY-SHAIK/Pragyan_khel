import React from 'react'
import { Link } from 'react-router-dom'
import { Focus, Github, Twitter, Linkedin, Mail } from 'lucide-react'

const Footer = () => {
  return (
    <footer className="border-t border-dark-600 bg-dark-800/50">
      <div className="max-w-7xl mx-auto px-6 py-12">
        <div className="grid grid-cols-1 md:grid-cols-4 gap-8">
          {/* Brand */}
          <div className="col-span-1 md:col-span-2">
            <Link to="/" className="flex items-center gap-3 mb-4">
              <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-primary to-accent flex items-center justify-center">
                <Focus className="w-6 h-6 text-white" />
              </div>
              <span className="text-2xl font-bold gradient-text">FocusAI</span>
            </Link>
            <p className="text-gray-400 max-w-md">
              Transform any camera into an intelligent cinematic tool with real-time 
              AI-powered object tracking and dynamic background blur.
            </p>
            <div className="flex gap-4 mt-6">
              <a href="#" className="text-gray-400 hover:text-accent transition-colors">
                <Github size={20} />
              </a>
              <a href="#" className="text-gray-400 hover:text-accent transition-colors">
                <Twitter size={20} />
              </a>
              <a href="#" className="text-gray-400 hover:text-accent transition-colors">
                <Linkedin size={20} />
              </a>
              <a href="#" className="text-gray-400 hover:text-accent transition-colors">
                <Mail size={20} />
              </a>
            </div>
          </div>

          {/* Links */}
          <div>
            <h4 className="font-semibold text-white mb-4">Product</h4>
            <ul className="space-y-2">
              <li><Link to="/demo" className="text-gray-400 hover:text-accent transition-colors">Demo</Link></li>
              <li><Link to="/features" className="text-gray-400 hover:text-accent transition-colors">Features</Link></li>
              <li><a href="#" className="text-gray-400 hover:text-accent transition-colors">Documentation</a></li>
              <li><a href="#" className="text-gray-400 hover:text-accent transition-colors">API</a></li>
            </ul>
          </div>

          <div>
            <h4 className="font-semibold text-white mb-4">Company</h4>
            <ul className="space-y-2">
              <li><Link to="/about" className="text-gray-400 hover:text-accent transition-colors">About</Link></li>
              <li><a href="#" className="text-gray-400 hover:text-accent transition-colors">Blog</a></li>
              <li><a href="#" className="text-gray-400 hover:text-accent transition-colors">Careers</a></li>
              <li><a href="#" className="text-gray-400 hover:text-accent transition-colors">Contact</a></li>
            </ul>
          </div>
        </div>

        <div className="border-t border-dark-600 mt-12 pt-8 flex flex-col md:flex-row justify-between items-center gap-4">
          <p className="text-gray-500 text-sm">
            © 2024 FocusAI. All rights reserved.
          </p>
          <div className="flex gap-6 text-sm">
            <a href="#" className="text-gray-500 hover:text-accent transition-colors">Privacy Policy</a>
            <a href="#" className="text-gray-500 hover:text-accent transition-colors">Terms of Service</a>
          </div>
        </div>
      </div>
    </footer>
  )
}

export default Footer
