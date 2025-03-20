"use client";

import useWebSocket, { ReadyState } from "react-use-websocket";
import { Card, CardContent, CardHeader, CardTitle } from "./ui/card";
import { useEffect, useRef, useState } from "react";
import { Button } from "./ui/button";
import { Alert, AlertDescription } from "./ui/alert";
import { toast } from "sonner";
import { Progress } from "./ui/progress";

type ResourceUsageTextColours =
  | "text-green-500"
  | "text-orange-500"
  | "text-red-500";

type ResourceUsageBackgroundColours =
  | "bg-green-500"
  | "bg-orange-500"
  | "bg-red-500";

type RamStats = {
  total: number;
  used: number;
};

export default function CameraStream() {
  const [isApplicationStarted, setIsApplicationStarted] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [notificationSound, setNotificationSound] =
    useState<HTMLAudioElement | null>(null);
  const [hailoDeviceTemperature, setHailoDeviceTemperature] = useState<
    string | null
  >(null);
  const [hailoDeviceTemperatureColor, setHailoDeviceTemperatureColor] =
    useState<ResourceUsageTextColours>("text-green-500");
  const [cpuUsage, setCPUUsage] = useState<string | null>(null);
  const [cpuUsageColor, setCPUUsageColor] =
    useState<ResourceUsageTextColours>("text-green-500");
  const [cpuProgressColour, setCpuProgressColour] =
    useState<ResourceUsageBackgroundColours>("bg-green-500");
  const [cpuTemperature, setCPUTemperature] = useState<string | null>(null);
  const [cpuTemperatureColor, setCPUTemperatureColor] =
    useState<ResourceUsageTextColours>("text-green-500");
  const [ramStats, setRamStats] = useState<RamStats | null>(null);
  const [ramStatsColour, setRamStatsColour] =
    useState<ResourceUsageTextColours>("text-green-500");
  const [ramProgressColour, setRamProgressColour] =
    useState<ResourceUsageBackgroundColours>("bg-green-500");

  useEffect(() => {
    setNotificationSound(new Audio("/notification_sound.wav"));
  }, []);

  const playNotificationSound = () => {
    if (notificationSound) {
      notificationSound
        .play()
        .catch((error) =>
          console.error("❌ 🔉 Error playing notification sound: ", error)
        );
    }
  };

  const { lastMessage, readyState } = useWebSocket("/cpp/ws", {
    shouldReconnect: () => true,
    onOpen: async () => console.log("Connected to web socket server"),
    onClose: () => console.log("Disconnected from web socket server"),
    onError: (error) => console.error("Error with web socket server: ", error),
    onMessage: (message) => {
      if (message.data instanceof Blob) return;

      try {
        const data = JSON.parse(message.data);

        if (data.message) {
          if (data.message === "application started") {
            setIsApplicationStarted(true);
            return;
          }

          if (data.message === "application stopped") {
            setIsApplicationStarted(false);
            return;
          }
        }

        if (data.event) {
          if (data.event === "hailo device temperature") {
            setHailoDeviceTemperature(data.temperature.toFixed(2));
            return;
          }

          if (data.event === "cpu stats") {
            setCPUUsage(data.usage.toFixed(2));
            setCPUTemperature(data.temperature.toFixed(2));
            return;
          }

          if (data.event === "ram stats") {
            setRamStats({ total: data.total, used: data.used });
            return;
          }

          if (data.event === "person detected") {
            playNotificationSound();
            toast("👤 Person Detected");
            return;
          }
        }
      } catch (error) {
        console.error("❌ Error parsing websocket message: ", error);
      }
    },
  });

  const connectionStatus = ReadyState[readyState];

  const imgRef = useRef<HTMLImageElement>(null);

  const toggleApplication = async () => {
    setError(null);
    const action = isApplicationStarted ? "stop" : "start";

    const response = await fetch(`/cpp/${action}`, { method: "POST" });

    if (!response.ok) {
      setError(`Failed to ${action} AI Camera application`);
      return;
    }

    setIsApplicationStarted(!isApplicationStarted);
  };

  useEffect(() => {
    if (imgRef.current?.src) {
      URL.revokeObjectURL(imgRef.current.src);
    }

    if (lastMessage?.data instanceof Blob) {
      const url = URL.createObjectURL(lastMessage.data);
      if (imgRef.current) {
        imgRef.current.src = url;
      }
    }
  }, [lastMessage]);

  useEffect(() => {
    if (cpuUsage) {
      const floatCpuUsage = parseFloat(cpuUsage);

      if (floatCpuUsage < 50) {
        setCPUUsageColor("text-green-500");
        setCpuProgressColour("bg-green-500");
      } else if (floatCpuUsage < 75) {
        setCPUUsageColor("text-orange-500");
        setCpuProgressColour("bg-orange-500");
      } else {
        setCPUUsageColor("text-red-500");
        setCpuProgressColour("bg-red-500");
      }
    }
  }, [cpuUsage]);

  useEffect(() => {
    if (hailoDeviceTemperature) {
      const floatHailoDeviceTemperature = parseFloat(hailoDeviceTemperature);

      if (floatHailoDeviceTemperature < 50) {
        setHailoDeviceTemperatureColor("text-green-500");
      } else if (floatHailoDeviceTemperature < 75) {
        setHailoDeviceTemperatureColor("text-orange-500");
      } else {
        setHailoDeviceTemperatureColor("text-red-500");
      }
    }
  }, [hailoDeviceTemperature]);

  useEffect(() => {
    if (cpuTemperature) {
      const floatCPUTemperature = parseFloat(cpuTemperature);

      if (floatCPUTemperature < 50) {
        setCPUTemperatureColor("text-green-500");
      } else if (floatCPUTemperature < 75) {
        setCPUTemperatureColor("text-orange-500");
      } else {
        setCPUTemperatureColor("text-red-500");
      }
    }
  }, [cpuTemperature]);

  useEffect(() => {
    if (ramStats) {
      const ramUsage = (ramStats.used / ramStats.total) * 100;

      if (ramUsage < 50) {
        setRamStatsColour("text-green-500");
        setRamProgressColour("bg-green-500");
      } else if (ramUsage < 75) {
        setRamStatsColour("text-orange-500");
        setRamProgressColour("bg-orange-500");
      } else {
        setRamStatsColour("text-red-500");
        setRamProgressColour("bg-red-500");
      }
    }
  }, [ramStats]);

  useEffect(() => {
    if (!isApplicationStarted) {
      setCPUTemperature(null);
      setCPUUsage(null);
      setHailoDeviceTemperature(null);
      setRamStats(null);
    }
  }, [isApplicationStarted]);

  return (
    <Card>
      <CardHeader>
        <CardTitle>AI Object Detection Stream</CardTitle>
      </CardHeader>

      <CardContent className="flex flex-col space-y-2.5 items-center">
        <div className="w-full flex justify-between sm:flex-row flex-col max-w-screen-2xl">
          <div className="space-y-4 mb-2">
            <p>Stream status: {connectionStatus}</p>

            <Button
              onClick={toggleApplication}
              variant={isApplicationStarted ? "destructive" : "default"}
            >
              {isApplicationStarted ? "Stop" : "Start"} Application
            </Button>
          </div>
          <div>
            {hailoDeviceTemperature && (
              <p>
                Hailo NPU temp:{" "}
                <span className={hailoDeviceTemperatureColor}>
                  {hailoDeviceTemperature} °C
                </span>
              </p>
            )}
            {cpuTemperature && (
              <p>
                CPU temp:{" "}
                <span className={cpuTemperatureColor}>{cpuTemperature} °C</span>
              </p>
            )}
            {cpuUsage && (
              <div className="flex gap-2 items-center">
                <p>CPU usage: </p>
                <Progress
                  value={Number(cpuUsage)}
                  className="w-32"
                  override_colour={cpuProgressColour}
                />
                <p className={`text-sm ${cpuUsageColor}`}>{cpuUsage} %</p>
              </div>
            )}
            {ramStats && (
              <div className="flex gap-2 items-center">
                <p>RAM usage: </p>
                <Progress
                  value={(ramStats.used / ramStats.total) * 100}
                  className="w-32"
                  override_colour={ramProgressColour}
                />
                <p className={`text-sm ${ramStatsColour}`}>
                  {(ramStats.used / (1024 * 1024)).toFixed(2)} /{" "}
                  {(ramStats.total / (1024 * 1024)).toFixed(2)} GB
                </p>
              </div>
            )}
          </div>
        </div>

        {error && (
          <Alert variant="destructive">
            <AlertDescription>{error}</AlertDescription>
          </Alert>
        )}

        <div className="relative bg-green-950 aspect-video w-full max-w-screen-2xl">
          {!isApplicationStarted && (
            <div className="absolute inset-0 flex items-center justify-center">
              <p>
                {!isApplicationStarted &&
                  "Press start to start the application."}
              </p>
            </div>
          )}

          {readyState === ReadyState.OPEN && isApplicationStarted && (
            <img
              alt="JPEG Stream"
              ref={imgRef}
              className="w-full h-full object-contain"
            />
          )}
        </div>
      </CardContent>
    </Card>
  );
}
