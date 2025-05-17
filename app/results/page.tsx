"use client"

import { useState, useEffect } from "react"
import { useRouter } from "next/navigation"
import { ArrowLeft, ArrowRight, Home, Layers, Zap } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { ScrollArea, ScrollBar } from "@/components/ui/scroll-area"
import { Card, CardContent } from "@/components/ui/card"

// Mock data for the step-by-step guide
const mockSteps = [
  {
    title: "Raw Materials",
    description:
      "The manufacturing process begins with the selection of high-quality raw materials, including aluminum alloy, steel components, and polymer composites.",
    image: "/placeholder.svg?height=300&width=400",
  },
  {
    title: "Casting Process",
    description:
      "The aluminum body is created through a precision die-casting process, where molten aluminum is injected into molds under high pressure.",
    image: "/placeholder.svg?height=300&width=400",
  },
  {
    title: "Machining",
    description:
      "CNC machines precisely cut and shape the cast components to exact specifications, creating mounting points and functional features.",
    image: "/placeholder.svg?height=300&width=400",
  },
  {
    title: "Assembly",
    description:
      "Components are assembled using automated systems and skilled technicians, with precision alignment of all moving parts.",
    image: "/placeholder.svg?height=300&width=400",
  },
  {
    title: "Quality Control",
    description:
      "Each unit undergoes rigorous testing for functionality, durability, and performance under various conditions.",
    image: "/placeholder.svg?height=300&width=400",
  },
  {
    title: "Finishing",
    description:
      "Surface treatments and coatings are applied to enhance durability, appearance, and resistance to environmental factors.",
    image: "/placeholder.svg?height=300&width=400",
  },
]

// Mock data for FEA analysis
const feaOptions = [
  { id: "stress", name: "Stress Analysis", color: "bg-red-500" },
  { id: "strain", name: "Strain Distribution", color: "bg-blue-500" },
  { id: "thermal", name: "Thermal Analysis", color: "bg-orange-500" },
  { id: "vibration", name: "Vibration Modes", color: "bg-purple-500" },
]

export default function ResultsPage() {
  const [capturedImage, setCapturedImage] = useState<string | null>(null)
  const [activeFeaOption, setActiveFeaOption] = useState<string | null>(null)
  const router = useRouter()

  useEffect(() => {
    // Retrieve the captured image from localStorage
    const storedImage = localStorage.getItem("capturedImage")
    if (storedImage) {
      setCapturedImage(storedImage)
    } else {
      // If no image is found, redirect to home
      router.push("/")
    }
  }, [router])

  return (
    <main className="flex min-h-screen flex-col items-center p-4 md:p-8">
      <div className="w-full max-w-7xl flex flex-col gap-6">
        <div className="flex justify-between items-center">
          <Button variant="outline" size="icon" onClick={() => router.push("/")}>
            <Home className="h-5 w-5" />
          </Button>
          <h1 className="text-2xl md:text-3xl font-bold">How It&apos;s Made</h1>
          <div className="w-10"></div> {/* Spacer for alignment */}
        </div>

        <Tabs defaultValue="steps" className="w-full">
          <TabsList className="grid w-full grid-cols-2">
            <TabsTrigger value="steps" className="flex items-center gap-2">
              <Layers className="h-4 w-4" />
              Step-by-Step
            </TabsTrigger>
            <TabsTrigger value="fea" className="flex items-center gap-2">
              <Zap className="h-4 w-4" />
              FEA Analysis
            </TabsTrigger>
          </TabsList>

          <TabsContent value="steps" className="mt-4">
            <div className="flex flex-col gap-4">
              <div className="w-full aspect-video md:aspect-[16/9] bg-muted rounded-lg overflow-hidden">
                {capturedImage && (
                  <img
                    src={capturedImage || "/placeholder.svg"}
                    alt="Analyzed object"
                    className="w-full h-full object-contain"
                  />
                )}
              </div>

              <div className="relative">
                <ScrollArea className="w-full whitespace-nowrap rounded-md border">
                  <div className="flex p-4 gap-4">
                    {mockSteps.map((step, index) => (
                      <Card key={index} className="w-[300px] flex-shrink-0">
                        <CardContent className="p-4">
                          <div className="aspect-[4/3] bg-muted rounded-md overflow-hidden mb-3">
                            <img
                              src={step.image || "/placeholder.svg"}
                              alt={step.title}
                              className="w-full h-full object-cover"
                            />
                          </div>
                          <h3 className="font-semibold text-lg">
                            {index + 1}. {step.title}
                          </h3>
                          <p className="text-sm text-muted-foreground whitespace-normal">{step.description}</p>
                        </CardContent>
                      </Card>
                    ))}
                  </div>
                  <ScrollBar orientation="horizontal" />
                </ScrollArea>

                <div className="absolute top-1/2 -translate-y-1/2 -left-4 hidden md:block">
                  <Button variant="outline" size="icon" className="rounded-full bg-background shadow-md">
                    <ArrowLeft className="h-4 w-4" />
                  </Button>
                </div>
                <div className="absolute top-1/2 -translate-y-1/2 -right-4 hidden md:block">
                  <Button variant="outline" size="icon" className="rounded-full bg-background shadow-md">
                    <ArrowRight className="h-4 w-4" />
                  </Button>
                </div>
              </div>
            </div>
          </TabsContent>

          <TabsContent value="fea" className="mt-4">
            <div className="flex flex-col gap-6">
              <div className="w-full aspect-video md:aspect-[16/9] bg-muted rounded-lg overflow-hidden relative">
                {capturedImage && (
                  <>
                    <img
                      src={capturedImage || "/placeholder.svg"}
                      alt="Analyzed object"
                      className="w-full h-full object-contain"
                    />
                    {activeFeaOption && (
                      <div className="absolute inset-0 bg-black/50 flex items-center justify-center">
                        <div className="w-full h-full max-w-[80%] max-h-[80%] relative">
                          <div
                            className={`absolute inset-0 opacity-50 ${
                              feaOptions.find((opt) => opt.id === activeFeaOption)?.color || "bg-red-500"
                            } blur-md rounded-lg`}
                          ></div>
                          <img
                            src="/placeholder.svg?height=400&width=600"
                            alt="FEA Visualization"
                            className="w-full h-full object-contain relative z-10"
                          />
                        </div>
                      </div>
                    )}
                  </>
                )}
              </div>

              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                {feaOptions.map((option) => (
                  <Button
                    key={option.id}
                    variant={activeFeaOption === option.id ? "default" : "outline"}
                    className="h-auto py-3"
                    onClick={() => setActiveFeaOption(option.id === activeFeaOption ? null : option.id)}
                  >
                    <div className="flex flex-col items-center gap-2">
                      <div className={`w-4 h-4 rounded-full ${option.color}`}></div>
                      <span>{option.name}</span>
                    </div>
                  </Button>
                ))}
              </div>

              <div className="bg-muted rounded-lg p-4">
                <h3 className="font-semibold mb-2">FEA Simulation Results</h3>
                {activeFeaOption ? (
                  <div className="space-y-2">
                    <p className="text-sm">
                      {activeFeaOption === "stress" &&
                        "The stress analysis shows maximum stress concentration at the connection points, with values within acceptable safety margins. The design demonstrates good load distribution across the structure."}
                      {activeFeaOption === "strain" &&
                        "Strain distribution indicates minimal deformation under normal operating conditions. The material elasticity provides sufficient recovery from applied forces."}
                      {activeFeaOption === "thermal" &&
                        "Thermal analysis reveals effective heat dissipation through the aluminum components. Critical electronic components remain within optimal temperature ranges during operation."}
                      {activeFeaOption === "vibration" &&
                        "Vibration analysis identifies natural frequencies and mode shapes. The design avoids resonance with common operational frequencies, ensuring stable performance."}
                    </p>
                    <div className="grid grid-cols-2 gap-4 mt-4">
                      <div>
                        <p className="text-xs text-muted-foreground mb-1">Maximum Value</p>
                        <p className="font-mono font-medium">
                          {activeFeaOption === "stress" && "287.5 MPa"}
                          {activeFeaOption === "strain" && "0.0032 mm/mm"}
                          {activeFeaOption === "thermal" && "72.4°C"}
                          {activeFeaOption === "vibration" && "124.7 Hz"}
                        </p>
                      </div>
                      <div>
                        <p className="text-xs text-muted-foreground mb-1">Safety Factor</p>
                        <p className="font-mono font-medium">
                          {activeFeaOption === "stress" && "1.74"}
                          {activeFeaOption === "strain" && "2.31"}
                          {activeFeaOption === "thermal" && "1.38"}
                          {activeFeaOption === "vibration" && "1.92"}
                        </p>
                      </div>
                    </div>
                  </div>
                ) : (
                  <p className="text-sm text-muted-foreground">
                    Select an analysis type above to view simulation results
                  </p>
                )}
              </div>
            </div>
          </TabsContent>
        </Tabs>
      </div>
    </main>
  )
}
