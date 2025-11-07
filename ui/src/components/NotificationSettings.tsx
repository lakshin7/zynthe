/**
 * NotificationSettings Component
 * UI controls for notification preferences
 */

import React from 'react';
import useNotifications from '../hooks/useNotifications';

const NotificationSettings: React.FC = () => {
  const { permission, soundEnabled, requestPermission, toggleSound, playNotificationSound } = useNotifications();

  const handleEnableNotifications = async () => {
    const result = await requestPermission();
    if (result === 'granted') {
      playNotificationSound(); // Test sound
    }
  };

  return (
    <div className="notification-settings">
      <h3 className="text-sm font-semibold text-gray-700 mb-3">
        🔔 Notifications
      </h3>
      
      <div className="space-y-3">
        {/* Browser Notification Permission */}
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-2">
            <span className="text-sm text-gray-600">
              Browser Notifications
            </span>
            {permission === 'granted' && (
              <span className="text-xs bg-green-100 text-green-700 px-2 py-0.5 rounded">
                Enabled
              </span>
            )}
            {permission === 'denied' && (
              <span className="text-xs bg-red-100 text-red-700 px-2 py-0.5 rounded">
                Blocked
              </span>
            )}
          </div>
          
          {permission === 'default' && (
            <button
              onClick={handleEnableNotifications}
              className="px-3 py-1 text-xs bg-blue-600 text-white rounded hover:bg-blue-700 transition"
            >
              Enable
            </button>
          )}
          
          {permission === 'denied' && (
            <span className="text-xs text-gray-500">
              Allow in browser settings
            </span>
          )}
        </div>

        {/* Sound Toggle */}
        <div className="flex items-center justify-between">
          <label htmlFor="sound-toggle" className="flex items-center space-x-2 cursor-pointer">
            <input
              id="sound-toggle"
              type="checkbox"
              checked={soundEnabled}
              onChange={(e) => toggleSound(e.target.checked)}
              className="w-4 h-4 text-blue-600 rounded focus:ring-2 focus:ring-blue-500"
            />
            <span className="text-sm text-gray-600">
              Play sound on completion
            </span>
          </label>
          
          <button
            onClick={playNotificationSound}
            disabled={!soundEnabled}
            className="px-3 py-1 text-xs bg-gray-200 text-gray-700 rounded hover:bg-gray-300 transition disabled:opacity-50 disabled:cursor-not-allowed"
            title="Test sound"
          >
            🔊 Test
          </button>
        </div>

        {/* Help Text */}
        <div className="text-xs text-gray-500 bg-gray-50 p-2 rounded">
          <p>
            <strong>Tip:</strong> Enable notifications to get alerts when training completes,
            even if you're working in another tab or application.
          </p>
        </div>
      </div>
    </div>
  );
};

export default NotificationSettings;
