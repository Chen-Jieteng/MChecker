import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'
import vueJsx from '@vitejs/plugin-vue-jsx'
import { fileURLToPath, URL } from 'node:url'

export default defineConfig({
  plugins: [vue(), vueJsx()],
  resolve: {
    alias: {
      '@': fileURLToPath(new URL('../douyin/src', import.meta.url)),
      '@home': fileURLToPath(new URL('./src', import.meta.url)),
      reporting: fileURLToPath(new URL('../reporting', import.meta.url)),
      marking: fileURLToPath(new URL('../marking', import.meta.url)),
      feedback: fileURLToPath(new URL('../feedback', import.meta.url)),
      'douyin': fileURLToPath(new URL('../douyin/src', import.meta.url))
    }
  },
  server: {
    host: '0.0.0.0',
    port: 3000,
    strictPort: true,
    proxy: {
      '/douyin': {
        target: 'http://127.0.0.1:3002',
        changeOrigin: true,
        ws: true,
        rewrite: (path) => path.replace(/^\/douyin/, '')
      }
    }
  }
})


