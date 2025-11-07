/**
 * useNotifications Hook
 * Handles browser notifications and sound alerts for training completion
 */

import { useEffect, useState } from 'react';

interface NotificationOptions {
  title: string;
  body: string;
  icon?: string;
  playSound?: boolean;
}

export const useNotifications = () => {
  const [permission, setPermission] = useState<NotificationPermission>('default');
  const [soundEnabled, setSoundEnabled] = useState(() => {
    // Load from localStorage
    const saved = localStorage.getItem('notificationSoundEnabled');
    return saved ? JSON.parse(saved) : true;
  });

  useEffect(() => {
    // Check current permission
    if ('Notification' in window) {
      setPermission(Notification.permission);
    }
  }, []);

  const requestPermission = async () => {
    if ('Notification' in window) {
      const result = await Notification.requestPermission();
      setPermission(result);
      return result;
    }
    return 'denied';
  };

  const playNotificationSound = () => {
    if (!soundEnabled) return;
    
    try {
      // Create audio context for notification sound
      const audioContext = new (window.AudioContext || (window as any).webkitAudioContext)();
      
      // Create a pleasant notification sound (two-tone chime)
      const playTone = (frequency: number, startTime: number, duration: number) => {
        const oscillator = audioContext.createOscillator();
        const gainNode = audioContext.createGain();
        
        oscillator.connect(gainNode);
        gainNode.connect(audioContext.destination);
        
        oscillator.frequency.value = frequency;
        oscillator.type = 'sine';
        
        // Envelope for smooth sound
        gainNode.gain.setValueAtTime(0, startTime);
        gainNode.gain.linearRampToValueAtTime(0.3, startTime + 0.01);
        gainNode.gain.exponentialRampToValueAtTime(0.01, startTime + duration);
        
        oscillator.start(startTime);
        oscillator.stop(startTime + duration);
      };
      
      const now = audioContext.currentTime;
      // Two-tone notification: C5 -> E5
      playTone(523.25, now, 0.15); // C5
      playTone(659.25, now + 0.15, 0.25); // E5
      
    } catch (error) {
      console.error('Failed to play notification sound:', error);
    }
  };

  const showNotification = async (options: NotificationOptions) => {
    const { title, body, icon, playSound = true } = options;

    // Play sound if enabled
    if (playSound && soundEnabled) {
      playNotificationSound();
    }

    // Show browser notification if permitted
    if ('Notification' in window && permission === 'granted') {
      try {
        const notification = new Notification(title, {
          body,
          icon: icon || '/zynthe-icon.png',
          badge: '/zynthe-icon.png',
          requireInteraction: false,
          silent: !soundEnabled, // Use silent mode if sound is disabled
        });

        // Auto-close after 5 seconds
        setTimeout(() => notification.close(), 5000);

        // Focus window when notification is clicked
        notification.onclick = () => {
          window.focus();
          notification.close();
        };

        return notification;
      } catch (error) {
        console.error('Failed to show notification:', error);
      }
    }
  };

  const toggleSound = (enabled: boolean) => {
    setSoundEnabled(enabled);
    localStorage.setItem('notificationSoundEnabled', JSON.stringify(enabled));
  };

  return {
    permission,
    soundEnabled,
    requestPermission,
    showNotification,
    toggleSound,
    playNotificationSound,
  };
};

export default useNotifications;
