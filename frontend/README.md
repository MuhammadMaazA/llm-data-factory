# LLM Data Factory - Frontend

A modern React frontend for the LLM Data Factory project, built with Vite, TypeScript, and shadcn/ui components.

## Getting Started

This frontend provides a web interface for the LLM Data Factory customer support ticket classifier.

### Prerequisites

- Node.js 18+ and npm (install with [nvm](https://github.com/nvm-sh/nvm#installing-and-updating))
- Backend API server running (see main project README)

### Installation

```sh
# Navigate to the frontend directory
cd frontend

# Install dependencies
npm install

# Start the development server
npm run dev
```

The frontend will be available at `http://localhost:5173`

### Building for Production

```sh
# Build the project
npm run build

# Preview the production build
npm run preview
```
## Technologies Used

This frontend is built with modern web technologies:

- **Vite** - Fast build tool and development server
- **TypeScript** - Type-safe JavaScript
- **React** - Component-based UI library
- **shadcn/ui** - Beautiful and accessible UI components
- **Tailwind CSS** - Utility-first CSS framework
- **React Router** - Client-side routing
- **React Query** - Data fetching and state management

## Project Structure

```
frontend/
├── src/
│   ├── components/
│   │   ├── landing/     # Landing page components
│   │   └── ui/          # Reusable UI components (shadcn/ui)
│   ├── pages/           # Page components
│   ├── hooks/           # Custom React hooks
│   ├── lib/             # Utility functions
│   └── main.tsx         # Application entry point
├── public/              # Static assets
└── package.json         # Dependencies and scripts
```

## API Integration

The frontend communicates with the Python backend API for:
- Ticket classification predictions
- Model evaluation results
- Synthetic data generation status

Ensure the backend server is running on `http://localhost:8000` (or update the API base URL in the configuration).
