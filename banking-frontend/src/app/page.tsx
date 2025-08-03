"use client";

import { useState, useRef, useEffect } from "react";
import axios from "axios";
import {
  FaUser,
  FaRobot,
  FaMicrophone,
  FaStop,
  FaPaperPlane,
} from "react-icons/fa";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import rehypeRaw from "rehype-raw";
import { useVoiceRecording } from "../hooks/useVoiceRecording";

interface Message {
  id: number;
  text: string;
  sender: "user" | "agent";
  timestamp: Date;
  type?: "text" | "voice";
}

export default function BankingChat() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [inputText, setInputText] = useState("");
  const [isTyping, setIsTyping] = useState(false);
  const [senderId, setSenderId] = useState<string>("");
  const [isClient, setIsClient] = useState(false);
  const [isScrolled, setIsScrolled] = useState(false);
  const [isProcessingVoice, setIsProcessingVoice] = useState(false);

  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  // Voice recording hook
  const { isRecording, startRecording, stopRecording, audioLevel } =
    useVoiceRecording();

  // ...existing useEffect code...
  useEffect(() => {
    setIsClient(true);
    setSenderId(
      `web-user-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`
    );

    setMessages([
      {
        id: 1,
        text: "Hello! I'm **Kora**, your **Banking AI Agent**. How can I help you today?",
        sender: "agent",
        timestamp: new Date(),
      },
    ]);
  }, []);

  useEffect(() => {
    const handleScroll = () => {
      setIsScrolled(window.scrollY > 50);
    };

    window.addEventListener("scroll", handleScroll);
    return () => window.removeEventListener("scroll", handleScroll);
  }, []);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, isTyping]);

  const sendToBackend = async (text: string): Promise<string> => {
    try {
      const res = await axios.post("http://localhost:8080/api/chat", {
        sender_id: senderId,
        message: text,
      });
      return res.data.reply;
    } catch (error) {
      console.error("Backend error:", error);
      throw new Error("Network error — is the backend running?");
    }
  };

  const sendVoiceToBackend = async (audioBlob: Blob): Promise<string> => {
    try {
      const formData = new FormData();
      formData.append("audio", audioBlob, "voice-message.webm");
      formData.append("sender_id", senderId);

      const res = await axios.post(
        "http://localhost:8080/api/voice",
        formData,
        {
          headers: {
            "Content-Type": "multipart/form-data",
          },
          timeout: 120000, // 120 second timeout for voice processing
        }
      );
      return res.data.reply;
    } catch (error) {
      console.error("Voice backend error:", error);
      throw new Error("Voice processing error — please try again");
    }
  };

  const handleSendMessage = async () => {
    if (!inputText.trim() || !senderId || !isClient) return;

    const userMsg: Message = {
      id: Date.now(),
      text: inputText,
      sender: "user",
      timestamp: new Date(),
      type: "text",
    };

    setMessages((prev) => [...prev, userMsg]);
    setInputText("");
    setIsTyping(true);

    try {
      const reply = await sendToBackend(inputText);
      const agentMsg: Message = {
        id: Date.now() + 1,
        text: reply,
        sender: "agent",
        timestamp: new Date(),
        type: "text",
      };
      setMessages((prev) => [...prev, agentMsg]);
    } catch (error) {
      const errMsg: Message = {
        id: Date.now() + 1,
        text: "Network error — is the backend running?",
        sender: "agent",
        timestamp: new Date(),
        type: "text",
      };
      setMessages((prev) => [...prev, errMsg]);
    } finally {
      setIsTyping(false);
    }
  };

  const handleVoiceMessage = async () => {
    if (isRecording) {
      // Stop recording and send
      setIsProcessingVoice(true);
      try {
        const audioBlob = await stopRecording();
        if (audioBlob) {
          // Add user voice message to chat
          const userMsg: Message = {
            id: Date.now(),
            text: "Voice message",
            sender: "user",
            timestamp: new Date(),
            type: "voice",
          };
          setMessages((prev) => [...prev, userMsg]);
          setIsTyping(true);

          // Send to backend
          const reply = await sendVoiceToBackend(audioBlob);
          const agentMsg: Message = {
            id: Date.now() + 1,
            text: reply,
            sender: "agent",
            timestamp: new Date(),
            type: "text",
          };
          setMessages((prev) => [...prev, agentMsg]);
        }
      } catch (error) {
        const errMsg: Message = {
          id: Date.now() + 1,
          text: "Voice processing error — please try typing your message",
          sender: "agent",
          timestamp: new Date(),
          type: "text",
        };
        setMessages((prev) => [...prev, errMsg]);
      } finally {
        setIsProcessingVoice(false);
        setIsTyping(false);
      }
    } else {
      // Start recording
      try {
        await startRecording();
      } catch (error) {
        console.error("Failed to start recording:", error);
        alert("Could not access microphone. Please check permissions.");
      }
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !isRecording) {
      handleSendMessage();
    }
  };

  const markdownComponents = {
    table: ({ children }: any) => (
      <div className="overflow-x-auto my-2 sm:my-4">
        <table className="min-w-full border-collapse border border-gray-300 bg-gray-50 rounded-lg text-xs sm:text-sm">
          {children}
        </table>
      </div>
    ),
    thead: ({ children }: any) => (
      <thead className="bg-gray-100">{children}</thead>
    ),
    th: ({ children }: any) => (
      <th className="border border-gray-300 px-2 sm:px-4 py-1 sm:py-2 text-left font-semibold text-gray-700 text-xs sm:text-sm">
        {children}
      </th>
    ),
    td: ({ children }: any) => (
      <td className="border border-gray-300 px-2 sm:px-4 py-1 sm:py-2 text-gray-600 text-xs sm:text-sm">
        {children}
      </td>
    ),
    tr: ({ children }: any) => <tr className="hover:bg-gray-50">{children}</tr>,
    p: ({ children }: any) => (
      <p className="mb-1 sm:mb-2 last:mb-0">{children}</p>
    ),
    strong: ({ children }: any) => (
      <strong className="font-semibold text-gray-900">{children}</strong>
    ),
    ul: ({ children }: any) => (
      <ul className="list-disc pl-4 sm:pl-5 mb-1 sm:mb-2 space-y-1">
        {children}
      </ul>
    ),
    ol: ({ children }: any) => (
      <ol className="list-decimal pl-4 sm:pl-5 mb-1 sm:mb-2 space-y-1">
        {children}
      </ol>
    ),
    li: ({ children }: any) => <li className="text-gray-700">{children}</li>,
    code: ({ children, className }: any) => {
      const isInline = !className;
      if (isInline) {
        return (
          <code className="bg-gray-100 px-1 py-0.5 rounded text-xs sm:text-sm font-mono text-gray-800">
            {children}
          </code>
        );
      }
      return (
        <pre className="bg-gray-100 p-2 sm:p-3 rounded-lg overflow-x-auto my-1 sm:my-2">
          <code className="text-xs sm:text-sm font-mono text-gray-800">
            {children}
          </code>
        </pre>
      );
    },
    blockquote: ({ children }: any) => (
      <blockquote className="border-l-4 border-gray-300 pl-3 sm:pl-4 italic text-gray-600 my-1 sm:my-2">
        {children}
      </blockquote>
    ),
  };

  if (!isClient) {
    return (
      <div
        className="min-h-screen flex flex-col"
        style={{ backgroundColor: "#eee" }}
      >
        {/* ...existing loading state... */}
        <header
          className={`fixed top-0 left-0 right-0 h-16 sm:h-20 shadow-md flex items-center justify-between px-3 sm:px-6 z-50 transition-all duration-300 ${
            isScrolled ? "bg-white/60 backdrop-blur-md" : "bg-white"
          }`}
        >
          <div className="flex items-center">
            <div className="w-20 h-20 sm:w-40 sm:h-40 flex items-center justify-center">
              <img
                src="/avz-logo.png"
                alt="Avanza Logo"
                className="w-20 h-20 sm:w-40 sm:h-40 object-contain"
              />
            </div>
          </div>
          <div className="absolute left-1/2 transform -translate-x-1/2">
            <h1
              className="text-lg sm:text-2xl font-semibold"
              style={{
                color: "#696969",
                fontFamily: "'Open Sans', sans-serif",
                letterSpacing: "0.025em",
              }}
            >
              <span className="hidden sm:inline">Kora AI Banking</span>
              <span className="sm:hidden">Kora AI</span>
            </h1>
          </div>
          <div className="w-8 sm:w-16"></div>
        </header>

        <div className="flex-1 flex items-center justify-center">
          <div className="text-center">
            <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-purple-600 mx-auto mb-4"></div>
            <p className="text-gray-600">Loading chat...</p>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div
      className="min-h-screen flex flex-col"
      style={{ backgroundColor: "#eee" }}
    >
      {/* ...existing header code... */}
      <header
        className={`fixed top-0 left-0 right-0 h-16 sm:h-20 shadow-md flex items-center justify-between px-3 sm:px-6 z-50 transition-all duration-300 ${
          isScrolled ? "bg-white/60 backdrop-blur-md" : "bg-white"
        }`}
      >
        <div className="flex items-center">
          <div className="w-20 h-20 sm:w-40 sm:h-40 flex items-center justify-center">
            <img
              src="/avz-logo.png"
              alt="Avanza Logo"
              className="w-20 h-20 sm:w-40 sm:h-40 object-contain"
            />
          </div>
        </div>
        <div className="absolute left-1/2 transform -translate-x-1/2">
          <h1
            className="text-lg sm:text-2xl font-semibold"
            style={{
              color: "#696969",
              fontFamily: "'Open Sans', sans-serif",
              letterSpacing: "0.025em",
            }}
          >
            <span className="hidden sm:inline">Kora AI Banking</span>
            <span className="sm:hidden">Kora AI</span>
          </h1>
        </div>
        <div className="w-8 sm:w-16"></div>
      </header>

      {/* Chat Messages */}
      <div className="flex-1 overflow-y-auto px-3 sm:px-4 py-4 sm:py-6 pt-20 sm:pt-24 pb-20 sm:pb-24">
        <div className="max-w-full sm:max-w-6xl mx-auto space-y-3 sm:space-y-4">
          {messages.map((message) => (
            <div
              key={message.id}
              className={`flex ${
                message.sender === "user" ? "justify-end" : "justify-start"
              } items-end space-x-2`}
            >
              {message.sender === "agent" && (
                <div
                  className="w-6 h-6 sm:w-8 sm:h-8 rounded-full flex items-center justify-center flex-shrink-0  border-gray-500"
                  style={{ backgroundColor: "#696969" }}
                >
                  <FaRobot className="w-3 h-3 sm:w-4 sm:h-4 text-white" />
                </div>
              )}

              <div
                className={`px-3 sm:px-4 py-2 sm:py-3 rounded-2xl shadow-sm ${
                  message.sender === "user"
                    ? " bg-white text-gray-800 border border-gray-500 text-left max-w-[85%] sm:max-w-2xl ml-auto"
                    : "bg-white text-gray-800 border border-gray-500 text-left max-w-[85%] sm:max-w-4xl mr-auto"
                }`}
                style={
                  message.sender === "user"
                    ? { backgroundColor: "#cecaca" }
                    : {}
                }
              >
                {message.sender === "user" ? (
                  <p
                    className={`text-xs sm:text-sm leading-relaxed text-left break-words ${
                      message.type === "voice"
                        ? "italic flex items-center gap-2"
                        : ""
                    }`}
                  >
                    {message.type === "voice" && (
                      <FaMicrophone className="w-3 h-3" />
                    )}
                    {message.text}
                  </p>
                ) : (
                  <div className="text-xs sm:text-sm leading-relaxed text-left prose prose-sm max-w-none">
                    <ReactMarkdown
                      remarkPlugins={[remarkGfm]}
                      rehypePlugins={[rehypeRaw]}
                      components={markdownComponents}
                    >
                      {message.text}
                    </ReactMarkdown>
                  </div>
                )}
                <span
                  className={`text-xs mt-1 block ${
                    message.sender === "user"
                      ? "text-gray-700 text-right"
                      : "text-gray-500 text-right"
                  }`}
                >
                  {message.timestamp.toLocaleTimeString([], {
                    hour: "2-digit",
                    minute: "2-digit",
                  })}
                </span>
              </div>

              {message.sender === "user" && (
                <div
                  className="w-6 h-6 sm:w-8 sm:h-8 rounded-full flex items-center justify-center flex-shrink-0 border border-gray-500"
                  style={{ backgroundColor: "#696969" }}
                >
                  <FaUser className="w-3 h-3 sm:w-4 sm:h-4 text-white" />
                </div>
              )}
            </div>
          ))}

          {(isTyping || isProcessingVoice) && (
            <div className="flex justify-start items-end space-x-2">
              <div
                className="w-6 h-6 sm:w-8 sm:h-8 rounded-full flex items-center justify-center flex-shrink-0 border  border-gray-500"
                style={{ backgroundColor: "#696969" }}
              >
                <FaRobot className="w-3 h-3 sm:w-4 sm:h-4 text-white" />
              </div>
              <div className="bg-white border border-gray-200 px-3 sm:px-4 py-2 sm:py-3 rounded-2xl max-w-[85%] sm:max-w-2xl mr-auto">
                <div className="flex space-x-1">
                  <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce"></div>
                  <div
                    className="w-2 h-2 bg-gray-400 rounded-full animate-bounce"
                    style={{ animationDelay: "0.1s" }}
                  ></div>
                  <div
                    className="w-2 h-2 bg-gray-400 rounded-full animate-bounce"
                    style={{ animationDelay: "0.2s" }}
                  ></div>
                </div>
              </div>
            </div>
          )}

          <div ref={messagesEndRef} />
        </div>
      </div>

      {/* Fixed Input Area with Voice Support */}
      <div
        className="fixed bottom-0 left-0 right-0 px-3 sm:px-4 py-3 sm:py-4 z-50"
        style={{ backgroundColor: "#eee" }}
      >
        <div className="max-w-full sm:max-w-4xl mx-auto">
          <div className="flex items-center space-x-2 sm:space-x-3 bg-white rounded-full px-3 sm:px-4 py-2 border border-gray-200">
            <input
              ref={inputRef}
              type="text"
              value={inputText}
              onChange={(e) => setInputText(e.target.value)}
              onKeyPress={handleKeyPress}
              placeholder={
                isRecording ? "Recording..." : "Type your message here..."
              }
              className="flex-1 px-3 sm:px-4 py-2 sm:py-3 text-sm sm:text-base border-0 rounded-full focus:outline-none transition-colors text-black bg-transparent"
              style={{ borderColor: inputText ? "#b32271" : "" }}
              onFocus={(e) => (e.target.style.borderColor = "#b32271")}
              onBlur={(e) =>
                (e.target.style.borderColor = inputText ? "#b32271" : "")
              }
              disabled={isRecording || isProcessingVoice}
            />
            {/* Voice Button */}
            <button
              onClick={handleVoiceMessage}
              disabled={isProcessingVoice}
              className={`w-10 h-10 sm:w-12 sm:h-12 rounded-full flex items-center justify-center transition-all duration-200 ease-in-out transform hover:scale-105 active:scale-95 ${
                isProcessingVoice ? "opacity-100 cursor-not-allowed" : ""
              }`}
              style={{
                backgroundColor: isRecording
                  ? "#b32271"
                  : isProcessingVoice
                  ? "#6b7280"
                  : "#6b7280", // gray-500 as default
              }}
              onMouseEnter={(e) => {
                if (!isRecording && !isProcessingVoice) {
                  e.currentTarget.style.backgroundColor = "#696969";
                  e.currentTarget.style.transform = "scale(1.05)"; // Scale up on hover
                }
              }}
              onMouseLeave={(e) => {
                if (!isRecording && !isProcessingVoice) {
                  e.currentTarget.style.backgroundColor = "#6b7280"; // back to gray-500
                  e.currentTarget.style.transform = "scale(1)"; // Back to normal scale
                }
              }}
              onMouseDown={(e) => {
                if (!isRecording && !isProcessingVoice) {
                  e.currentTarget.style.transform = "scale(0.95)"; // Scale down on click
                  e.currentTarget.style.backgroundColor = "#4b5563"; // Even darker on click
                }
              }}
              onMouseUp={(e) => {
                if (!isRecording && !isProcessingVoice) {
                  e.currentTarget.style.transform = "scale(1.05)"; // Back to hover scale
                  e.currentTarget.style.backgroundColor = "#696969"; // Back to hover color
                }
              }}
            >
              {isRecording ? (
                <FaStop className="w-4 h-4 sm:w-5 sm:h-5 text-white transition-transform duration-150" />
              ) : (
                <FaMicrophone className="w-4 h-4 sm:w-5 sm:h-5 text-white transition-transform duration-150" />
              )}
            </button>
            {/* Send Button */}
            <button
              onClick={handleSendMessage}
              disabled={!inputText.trim() || isRecording || isProcessingVoice}
              className="w-10 h-10 sm:w-12 sm:h-12 text-white rounded-full flex items-center justify-center disabled:bg-gray-500 disabled:cursor-not-allowed transition-all duration-200 ease-in-out transform hover:scale-105 active:scale-95"
              style={{
                backgroundColor: "#6b7280", // Always gray-500 as default
              }}
              onMouseEnter={(e) => {
                if (inputText.trim() && !isRecording && !isProcessingVoice) {
                  e.currentTarget.style.backgroundColor = "#696969"; // Darker on hover when enabled
                  e.currentTarget.style.transform = "scale(1.05)"; // Slight scale up
                }
              }}
              onMouseLeave={(e) => {
                // Always return to gray-500 and normal scale when mouse leaves
                e.currentTarget.style.backgroundColor = "#6b7280";
                e.currentTarget.style.transform = "scale(1)";
              }}
              onMouseDown={(e) => {
                if (inputText.trim() && !isRecording && !isProcessingVoice) {
                  e.currentTarget.style.transform = "scale(0.95)"; // Scale down on click
                  e.currentTarget.style.backgroundColor = "#4b5563"; // Even darker on click
                }
              }}
              onMouseUp={(e) => {
                if (inputText.trim() && !isRecording && !isProcessingVoice) {
                  e.currentTarget.style.transform = "scale(1.05)"; // Back to hover scale
                  e.currentTarget.style.backgroundColor = "#696969"; // Back to hover color
                }
              }}
            >
              <FaPaperPlane className="w-4 h-4 sm:w-5 sm:h-5 transition-transform duration-150" />
            </button>
          </div>

          {/* Recording indicator */}
          {isRecording && (
            <div className="mt-2 flex items-center justify-center space-x-2">
              <div className="flex items-center space-x-2">
                <div className="w-2 h-2 bg-gray-500 rounded-full animate-pulse"></div>
                <span className="text-sm text-gray-600">Recording...</span>
              </div>
              <div className="flex space-x-1">
                {Array.from({ length: 10 }).map((_, i) => (
                  <div
                    key={i}
                    className="w-1 bg-gray-400 rounded-full transition-all duration-100"
                    style={{
                      height: `${Math.max(
                        4,
                        (audioLevel / 255) * 20 + Math.random() * 5
                      )}px`,
                    }}
                  />
                ))}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
