'use client'

import { useRef, useState } from 'react'
import { ReactSketchCanvas, ReactSketchCanvasRef } from 'react-sketch-canvas'

export default function Home() {
  const canvasRef = useRef<ReactSketchCanvasRef>(null)
  const [prediction, setPrediction] = useState<{ prediction: number; confidence: number } | null>(null)

  const handlePredict = async () => {
    if (!canvasRef.current) return

    const base64Image = await canvasRef.current.exportImage('png')
    const res = await fetch(base64Image)
    const blob = await res.blob()
    const formData = new FormData()
    formData.append('image', blob, 'digit.png')

    try {
      const response = await fetch('http://127.0.0.1:5000/predict', {
        method: 'POST',
        body: formData,
      })
      const data = await response.json()

      setPrediction(data)
    } catch (error) {
      console.error('Error:', error)
    }
  }

  const handleClear = () => {
    canvasRef.current?.clearCanvas()
    setPrediction(null)
  }

  return (
    <div style={{ textAlign: 'center', marginTop: '50px', fontFamily: 'Arial', color: 'black' }}>
      <h1>Draw a Digit</h1>

      <div style={{ display: 'inline-block', border: '2px solid #333' }}>
        <ReactSketchCanvas
          ref={canvasRef}
          strokeWidth={15}
          strokeColor="white"
          canvasColor="black"
          width="280px"
          height="280px"
        />
      </div>

      <div style={{ marginTop: '20px' }}>
        <button onClick={handlePredict} style={{ padding: '10px 20px', marginRight: '10px', background: 'green', color: 'white' }}>
          PREDICT
        </button>
        <button onClick={handleClear} style={{ padding: '10px 20px', background: 'red', color: 'white' }}>
          CLEAR
        </button>
      </div>

      {prediction && (
        <div style={{ marginTop: '30px', color: 'white' }}>
          <h2>I see a: {prediction.prediction}</h2>
          <p>Confidence: {prediction.confidence.toFixed(1)}%</p>
        </div>
      )}
    </div>
  )
}