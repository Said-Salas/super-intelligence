'use client'

import { useRef, useState } from 'react'
import { ReactSketchCanvas, ReactSketchCanvasRef } from 'react-sketch-canvas'

export default function Home() {
  const canvasRef = useRef<ReactSketchCanvasRef>(null)
  const [prediction, setPrediction] = useState<{prediction: number; confidence: number} | null>(null)
  
  const handlePredict = async () => {
    if (!canvasRef.current) return
    const base64Image = await canvasRef.current.exportImage('png')
    
  }

  return (
    <div></div>
  );
}
