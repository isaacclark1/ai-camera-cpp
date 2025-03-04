"use client";

import useWebSocket, { ReadyState } from "react-use-websocket";
import { Card, CardContent, CardHeader, CardTitle } from "./ui/card";
import { useEffect, useRef, useState } from "react";
import { Button } from "./ui/button";
import { Alert, AlertDescription } from "./ui/alert";

export default function CameraStream() {
  const [isStreaming, setIsStreaming] = useState(false);

  const [error, setError] = useState<string | null>(null);

  const { lastMessage, readyState } = useWebSocket("/cpp/ws", {
    shouldReconnect: () => true,
  });

  const connectionStatus = ReadyState[readyState];

  const imgRef = useRef<HTMLImageElement>(null);

  const toggleStream = async () => {
    const action = isStreaming ? "stop" : "start";

    const response = await fetch(`/cpp/${action}`, { method: "POST" });

    if (!response.ok) {
      setError(`Failed to ${action} camera stream`);
      return;
    }

    setIsStreaming(!isStreaming);
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

  return (
    <Card>
      <CardHeader>
        <CardTitle>Camera Stream</CardTitle>
      </CardHeader>

      <CardContent className="flex flex-col space-y-2.5 items-center">
        <div className="space-y-4 self-start">
          <p>Stream status: {connectionStatus}</p>
          <Button
            onClick={toggleStream}
            variant={isStreaming ? "destructive" : "default"}
          >
            {isStreaming ? "Stop" : "Start"}
          </Button>
        </div>

        {error && (
          <Alert variant="destructive">
            <AlertDescription>{error}</AlertDescription>
          </Alert>
        )}

        <div className="relative bg-green-950 aspect-video w-full">
          {!isStreaming && (
            <div className="absolute inset-0 flex items-center justify-center">
              <p>{!isStreaming && "Press start to start streaming."}</p>
            </div>
          )}

          {readyState === ReadyState.OPEN && isStreaming && (
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
