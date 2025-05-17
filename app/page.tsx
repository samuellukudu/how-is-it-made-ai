import Link from "next/link"
import { Camera, Upload } from "lucide-react"
import { Button } from "@/components/ui/button"

export default function HomePage() {
  return (
    <main className="flex min-h-screen flex-col items-center justify-between p-4 md:p-24">
      <div className="w-full max-w-5xl flex flex-col items-center justify-center gap-8">
        <div className="text-center space-y-4">
          <h1 className="text-4xl md:text-6xl font-bold tracking-tight">How It&apos;s Made</h1>
          <p className="text-xl text-muted-foreground max-w-2xl">
            Discover how things are made with step-by-step guides and engineering insights
          </p>
        </div>

        <div className="w-full max-w-md aspect-video bg-muted rounded-lg flex items-center justify-center border-2 border-dashed border-muted-foreground/25">
          <p className="text-muted-foreground">Upload or take a photo to get started</p>
        </div>

        <div className="flex flex-col sm:flex-row gap-4 w-full max-w-md">
          <Link href="/camera" className="flex-1">
            <Button variant="default" size="lg" className="w-full gap-2">
              <Camera className="h-5 w-5" />
              Take Photo
            </Button>
          </Link>
          <Link href="/upload" className="flex-1">
            <Button variant="outline" size="lg" className="w-full gap-2">
              <Upload className="h-5 w-5" />
              Upload Photo
            </Button>
          </Link>
        </div>

        <div className="mt-8 text-center">
          <h2 className="text-xl font-semibold mb-2">How it works</h2>
          <ol className="text-muted-foreground text-left space-y-2 list-decimal list-inside">
            <li>Take a photo or upload an image of any object</li>
            <li>Our AI analyzes the object and creates a step-by-step breakdown</li>
            <li>Explore how it&apos;s made with interactive guides</li>
            <li>Generate engineering simulations with FEA analysis</li>
          </ol>
        </div>
      </div>
    </main>
  )
}
