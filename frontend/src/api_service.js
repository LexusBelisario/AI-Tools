// frontend/src/api.js
import { API_URL } from './config.js';

// Use relative path for local development, full URL for production
const API_BASE = process.env.NODE_ENV === 'development' ? "/api" : `${API_URL}/api`;

class ApiService {
  static getAuthToken() {
    // Check both token names for compatibility
    return localStorage.getItem('access_token') || localStorage.getItem('accessToken');
  }

  static getHeaders() {
    const token = this.getAuthToken();
    const headers = {
      'Content-Type': 'application/json',
    };
    
    if (token) {
      headers['Authorization'] = `Bearer ${token}`;
    }
    
    return headers;
  }

  static async handleResponse(response) {
    console.log(`Response status: ${response.status} for ${response.url}`);
    
    if (response.status === 401) {
      // Clear both possible token names
      localStorage.removeItem('access_token');
      localStorage.removeItem('accessToken');
      window.location.href = '/login';
      return null;
    }

    if (!response.ok) {
      const errorText = await response.text();
      console.error('API Error:', errorText);
      throw new Error(errorText || `HTTP error! status: ${response.status}`);
    }

    const text = await response.text();
    if (!text) {
      return null;
    }

    try {
      return JSON.parse(text);
    } catch (e) {
      console.error('Failed to parse JSON:', e);
      return text;
    }
  }

  static async get(endpoint) {
    const url = `${API_BASE}${endpoint}`;
    console.log(`GET request to: ${url}`);
    
    try {
      const response = await fetch(url, {
        method: 'GET',
        headers: this.getHeaders(),
      });
      return await this.handleResponse(response);
    } catch (error) {
      console.error('GET request failed:', error);
      throw error;
    }
  }

  static async post(endpoint, data) {
    const url = `${API_BASE}${endpoint}`;
    console.log(`POST request to: ${url}`, data);
    
    try {
      const response = await fetch(url, {
        method: 'POST',
        headers: this.getHeaders(),
        body: JSON.stringify(data),
      });
      return await this.handleResponse(response);
    } catch (error) {
      console.error('POST request failed:', error);
      throw error;
    }
  }

  static async put(endpoint, data) {
    const url = `${API_BASE}${endpoint}`;
    
    try {
      const response = await fetch(url, {
        method: 'PUT',
        headers: this.getHeaders(),
        body: JSON.stringify(data),
      });
      return await this.handleResponse(response);
    } catch (error) {
      console.error('PUT request failed:', error);
      throw error;
    }
  }

  static async delete(endpoint) {
    const url = `${API_BASE}${endpoint}`;
    
    try {
      const response = await fetch(url, {
        method: 'DELETE',
        headers: this.getHeaders(),
      });
      return await this.handleResponse(response);
    } catch (error) {
      console.error('DELETE request failed:', error);
      throw error;
    }
  }
}

// Export both the API_BASE and the ApiService
export default API_BASE;
export { ApiService, API_BASE as API };