"use client";

import { useState, useRef, useEffect } from "react";
import axios from "axios";
import { FaUser, FaRobot } from "react-icons/fa";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import rehypeRaw from "rehype-raw";

interface Message {
  id: number;
  text: string;
  sender: "user" | "agent";
  timestamp: Date;
}

export default function BankingChat() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [inputText, setInputText] = useState("");
  const [isTyping, setIsTyping] = useState(false);
  const [senderId, setSenderId] = useState<string>("");
  const [isClient, setIsClient] = useState(false);
  const [isScrolled, setIsScrolled] = useState(false);

  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    setIsClient(true);
    setSenderId(
      `web-user-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`
    );

    // Add initial message after hydration
    setMessages([
      {
        id: 1,
        text: "Hello! I'm **Kora**, your **Banking AI Agent**. How can I help you today?",
        sender: "agent",
        timestamp: new Date(),
      },
    ]);
  }, []);

  // Separate useEffect for scroll handling
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
      // Use your existing chat_api.py endpoint
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

  const handleSendMessage = async () => {
    if (!inputText.trim() || !senderId || !isClient) return;

    const userMsg: Message = {
      id: Date.now(),
      text: inputText,
      sender: "user",
      timestamp: new Date(),
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
      };
      setMessages((prev) => [...prev, agentMsg]);
    } catch (error) {
      const errMsg: Message = {
        id: Date.now() + 1,
        text: "Network error — is the backend running?",
        sender: "agent",
        timestamp: new Date(),
      };
      setMessages((prev) => [...prev, errMsg]);
    } finally {
      setIsTyping(false);
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === "Enter") {
      handleSendMessage();
    }
  };

  // Custom components for markdown rendering
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
    p: ({ children }: any) => <p className="mb-1 sm:mb-2 last:mb-0">{children}</p>,
    strong: ({ children }: any) => (
      <strong className="font-semibold text-gray-900">{children}</strong>
    ),
    ul: ({ children }: any) => (
      <ul className="list-disc pl-4 sm:pl-5 mb-1 sm:mb-2 space-y-1">{children}</ul>
    ),
    ol: ({ children }: any) => (
      <ol className="list-decimal pl-4 sm:pl-5 mb-1 sm:mb-2 space-y-1">{children}</ol>
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
          <code className="text-xs sm:text-sm font-mono text-gray-800">{children}</code>
        </pre>
      );
    },
    blockquote: ({ children }: any) => (
      <blockquote className="border-l-4 border-gray-300 pl-3 sm:pl-4 italic text-gray-600 my-1 sm:my-2">
        {children}
      </blockquote>
    ),
  };

  // Don't render interactive elements until client-side hydration is complete
  if (!isClient) {
    return (
      <div className="min-h-screen flex flex-col" style={{ backgroundColor: "#eee" }}>
        {/* Fixed Header */}
        <header
          className={`fixed top-0 left-0 right-0 h-16 sm:h-20 shadow-md flex items-center justify-between px-3 sm:px-6 z-50 transition-all duration-300 ${
            isScrolled ? "bg-white/80 backdrop-blur-md" : "bg-white"
          }`}
        >
          {/* Logo on the left */}
          <div className="flex items-center">
            <div className="w-20 h-20 sm:w-40 sm:h-40 flex items-center justify-center">
              <img
                src="/avz-logo.png"
                alt="Avanza Logo"
                className="w-20 h-20 sm:w-40 sm:h-40 object-contain"
              />
            </div>
          </div>
          {/* Centered title */}
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

          {/* Empty div to balance the layout */}
          <div className="w-8 sm:w-16"></div>
        </header>

        {/* Loading state */}
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
    <div className="min-h-screen flex flex-col" style={{ backgroundColor: "#eee" }}>
        {/* Fixed Header */}
        <header
          className={`fixed top-0 left-0 right-0 h-16 sm:h-20 shadow-md flex items-center justify-between px-3 sm:px-6 z-50 transition-all duration-300 ${
            isScrolled ? "bg-white/80 backdrop-blur-md" : "bg-white"
          }`}
        >
          {/* Logo on the left */}
          <div className="flex items-center">
            <div className="w-20 h-20 sm:w-40 sm:h-40 flex items-center justify-center">
              <img
                src="/avz-logo.png"
                alt="Avanza Logo"
                className="w-20 h-20 sm:w-40 sm:h-40 object-contain"
              />
            </div>
          </div>
          {/* Centered title */}
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

          {/* Empty div to balance the layout */}
          <div className="w-8 sm:w-16"></div>
        </header>

      {/* Chat Messages with top and bottom padding for fixed elements */}
      <div className="flex-1 overflow-y-auto px-3 sm:px-4 py-4 sm:py-6 pt-20 sm:pt-24 pb-20 sm:pb-24">
        <div className="max-w-full sm:max-w-6xl mx-auto space-y-3 sm:space-y-4">
          {messages.map((message) => (
            <div
              key={message.id}
              className={`flex ${
                message.sender === "user" ? "justify-end" : "justify-start"
              } items-end space-x-2`}
            >
              {/* Bot Avatar - only show for agent messages */}
              {message.sender === "agent" && (
                <div
                  className="w-6 h-6 sm:w-8 sm:h-8 rounded-full flex items-center justify-center flex-shrink-0 border-gray-400"
                  style={{ backgroundColor: "#696969" }}
                >
                  <FaRobot className="w-3 h-3 sm:w-4 sm:h-4 text-white" />
                </div>
              )}

              <div
                className={`px-3 sm:px-4 py-2 sm:py-3 rounded-2xl shadow-sm ${
                  message.sender === "user"
                    ? "text-white border border-gray-200 text-right max-w-[85%] sm:max-w-2xl ml-auto"
                    : "bg-white text-gray-800 border border-gray-200 text-left max-w-[85%] sm:max-w-4xl mr-auto"
                }`}
                style={
                  message.sender === "user"
                    ? { backgroundColor: "#696969" }
                    : {}
                }
              >
                {message.sender === "user" ? (
                  <p className="text-xs sm:text-sm leading-relaxed text-right break-words">
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
                      ? "text-gray-100 text-right"
                      : "text-gray-500 text-left"
                  }`}
                >
                  {message.timestamp.toLocaleTimeString([], {
                    hour: "2-digit",
                    minute: "2-digit",
                  })}
                </span>
              </div>

              {/* User Avatar - only show for user messages */}
              {message.sender === "user" && (
                <div
                  className="w-6 h-6 sm:w-8 sm:h-8 rounded-full flex items-center justify-center flex-shrink-0 border border-gray-400"
                  style={{ backgroundColor: "#696969" }}
                >
                  <FaUser className="w-3 h-3 sm:w-4 sm:h-4 text-white" />
                </div>
              )}
            </div>
          ))}

          {isTyping && (
            <div className="flex justify-start items-end space-x-2">
              <div
                className="w-6 h-6 sm:w-8 sm:h-8 rounded-full flex items-center justify-center flex-shrink-0 border border-gray-400"
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

      {/* Fixed Input Area */}
      <div className="fixed bottom-0 left-0 right-0 bg-white px-3 sm:px-4 py-3 sm:py-4 z-50 border-t border-gray-200">
        <div className="max-w-full sm:max-w-4xl mx-auto">
          <div className="flex items-center space-x-2 sm:space-x-3">
            <input
              ref={inputRef}
              type="text"
              value={inputText}
              onChange={(e) => setInputText(e.target.value)}
              onKeyPress={handleKeyPress}
              placeholder="Type your message here..."
              className="flex-1 px-3 sm:px-4 py-2 sm:py-3 text-sm sm:text-base border border-gray-500 rounded-full focus:outline-none transition-colors text-black"
              style={{ borderColor: inputText ? "#b11e6c" : "" }}
              onFocus={(e) => (e.target.style.borderColor = "#b11e6c")}
              onBlur={(e) =>
                (e.target.style.borderColor = inputText ? "#b11e6c" : "")
              }
            />
            <button
              onClick={handleSendMessage}
              disabled={!inputText.trim()}
              className="w-10 h-10 sm:w-12 sm:h-12 text-white rounded-full flex items-center justify-center disabled:bg-gray-400 border-b-black disabled:cursor-not-allowed transition-colors"
              style={{ backgroundColor: !inputText.trim() ? "" : "#696969" }}
              onMouseEnter={(e) => {
                if (inputText.trim()) {
                  e.currentTarget.style.backgroundColor = "#5a5a5a";
                }
              }}
              onMouseLeave={(e) => {
                if (inputText.trim()) {
                  e.currentTarget.style.backgroundColor = "#696969";
                }
              }}
            >
              <svg
                className="w-4 h-4 sm:w-5 sm:h-5"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8"
                />
              </svg>
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}