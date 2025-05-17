"use client"

import { useState, useRef, useCallback } from "react"
import { useRouter } from "next/navigation"
import Webcam from "react-webcam"
import { Camera, RotateCcw } from "lucide-react"
import { Button } from "@/components/ui/button"

export default function CameraPage() {
  const webcamRef = useRef<Webcam>(null)
  const [capturedImage, setCapturedImage] = useState<string | null>(null)
  const router = useRouter()

  const capture = useCallback(() => {
    const imageSrc = webcamRef.current?.getScreenshot()
    if (imageSrc) {
      setCapturedImage(imageSrc)
    }
  }, [webcamRef])

  const retake = () => {
    setCapturedImage(null)
  }

  const analyze = () => {
    if (capturedImage) {
      // In a real app, we would send the image to an API for analysis
      // For now, we'll just navigate to the results page
      localStorage.setItem("capturedImage", capturedImage)
      router.push("/results")
    }
  }

  return (
    <main className="flex min-h-screen flex-col items-center p-4 md:p-24">
      <div className="w-full max-w-2xl flex flex-col items-center gap-6">
        <h1 className="text-3xl font-bold">Take a Photo</h1>

        <div className="w-full aspect-video bg-black rounded-lg overflow-hidden">
          {!capturedImage ? (
            <Webcam
              audio={false}
              ref={webcamRef}
              screenshotFormat="image/jpeg"
              videoConstraints={{ facingMode: "environment" }}
              className="w-full h-full object-cover"
            />
          ) : (
            <img src={capturedImage || "/placeholder.svg"} alt="Captured" className="w-full h-full object-cover" />
          )}
        </div>

        <div className="flex gap-4 w-full">
          {!capturedImage ? (
            <Button onClick={capture} className="flex-1 gap-2">
              <Camera className="h-5 w-5" />
              Capture
            </Button>
          ) : (
            <>
              <Button variant="outline" onClick={retake} className="flex-1 gap-2">
                <RotateCcw className="h-5 w-5" />
                Retake
              </Button>
              <Button onClick={analyze} className="flex-1">
                Analyze
              </Button>
            </>
          )}
        </div>

        <Button variant="link" onClick={() => router.push("/")} className="mt-4">
          Back to Home
        </Button>
      </div>
    </main>
  )
}
