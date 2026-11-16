import React from 'react';
import { Bot, UserRound } from 'lucide-react';

/** The assistant's round robot avatar, shown beside every bot message and the typing indicator. */
export const BotAvatar: React.FC<{ size?: 'sm' | 'md' }> = ({ size = 'md' }) => (
  <div
    className={`${size === 'sm' ? 'w-8 h-8' : 'w-10 h-10'} shrink-0 rounded-full bg-gradient-to-br from-brand-500 to-brand-700 text-white flex items-center justify-center shadow-bubble ring-4 ring-white`}
    aria-hidden="true"
  >
    <Bot className={size === 'sm' ? 'w-4 h-4' : 'w-5 h-5'} strokeWidth={2.2} />
  </div>
);

/** The staff member's avatar, shown beside their own messages. */
export const UserAvatar: React.FC = () => (
  <div
    className="w-10 h-10 shrink-0 rounded-full bg-white text-emerald-600 flex items-center justify-center shadow-bubble ring-1 ring-brand-100"
    aria-hidden="true"
  >
    <UserRound className="w-5 h-5" strokeWidth={2.2} />
  </div>
);
