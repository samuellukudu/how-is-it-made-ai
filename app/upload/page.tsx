"use client"

import type React from "react"

import { useState, useCallback } from "react"
import { useRouter } from "next/navigation"
import { Upload, X } from "lucide-react"
import { Button } from "@/components/ui/button"

export default function UploadPage() {
  const [uploadedImage, setUploadedImage] = useState<string | null>(null)
  const router = useRouter()

  const handleFileChange = useCallback((event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0]
    if (file) {
      const reader = new FileReader()
      reader.onload = (e) => {
        setUploadedImage(e.target?.result as string)
      }
      reader.readAsDataURL(file)
    }
  }, [])

  const clearImage = () => {
    setUploadedImage(null)
  }

  const analyze = () => {
    if (uploadedImage) {
      // In a real app, we would send the image to an API for analysis
      // For now, we'll just navigate to the results page
      localStorage.setItem("capturedImage", uploadedImage)
      router.push("/results")
    }
  }

  return (
    <main className="flex min-h-screen flex-col items-center p-4 md:p-24">
      <div className="w-full max-w-2xl flex flex-col items-center gap-6">
        <h1 className="text-3xl font-bold">Upload a Photo</h1>

        <div className="w-full aspect-video bg-muted rounded-lg overflow-hidden border-2 border-dashed border-muted-foreground/25 flex items-center justify-center relative">
          {!uploadedImage ? (
            <label className="flex flex-col items-center justify-center cursor-pointer w-full h-full">
              <Upload className="h-10 w-10 text-muted-foreground mb-2" />
              <span className="text-muted-foreground">Click to upload an image</span>
              <input type="file" accept="image/*" className="hidden" onChange={handleFileChange} />
            </label>
          ) : (
            <>
              <img src={uploadedImage || "/placeholder.svg"} alt="Uploaded" className="w-full h-full object-cover" />
              <Button
                variant="destructive"
                size="icon"
                className="absolute top-2 right-2 rounded-full"
                onClick={clearImage}
              >
                <X className="h-4 w-4" />
              </Button>
            </>
          )}
        </div>

        <div className="flex gap-4 w-full">
          {uploadedImage && (
            <Button onClick={analyze} className="flex-1">
              Analyze
            </Button>
          )}
        </div>

        <Button variant="link" onClick={() => router.push("/")} className="mt-4">
          Back to Home
        </Button>
      </div>
    </main>
  )
}
