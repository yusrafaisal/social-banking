// src/lib/api.ts
const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8080';

export const apiConfig = {
  chat: `${API_BASE_URL}/api/chat`,
};