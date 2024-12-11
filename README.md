# Stakeholder Mapping Tool

This tool is a website built on Next.js and React, intended to identify stakeholders from user-input documents and generate a graph of coalitions between stakeholders.

For full functionality, please follow the steps below. You will need to have both the frontend server running at `127.0.0.1:3000` and the backend server running at `127.0.0.1:5000` for full functionality. Note that using `localhost` instead of `127.0.0.1` would occasionally cause errors on Mac during testing.

## Running the Frontend Locally

### Basic Next.js Setup
Enter into the frontend folder.
```bash
cd frontend
```

First, make sure you have `npm` installed (link [here](https://docs.npmjs.com/downloading-and-installing-node-js-and-npm)). Then, install the node packages locally by running the following command within the project root.
```bash
npm install
```

Once packages are installed, run the development server:
```bash
npm run dev
```

Open [http://127.0.0.1:3000](http://127.0.0.1:3000) with your browser to see the result. Making any changes in code and saving the file should update the localhost instance automatically. You can stop the server with Ctrl-C within the running terminal.

### Environment Variables
At the top of the frontend folder (sibling to `src/`), create a file called `.env.local`, which will hold environment secrets. Add two lines for your OpenAI API key and the backend URL. You may need to restart the development server to reload the environment variables.
```bash
OPENAI_API_KEY=<your key here>
NEXT_PUBLIC_RL_API_ENDPOINT=http://127.0.0.1:5000/api/rl-endpoint
```

## Running the Backend Server
For full functionality, you will also need to run the backend server. 

### Installing Packages
First, create a virtual environment to store the Python packages. You can use the `deactivate` command to exit the venv. If a virtual environment exists, you can simply activate it again.
```bash
# Mac:
python3 -m venv .venv               # Create the venv named .venv
source .venv/bin/activate           # Enter into the virtual environment
pip install -r requirements-mac.txt # Install packages

# Windows (cmd prompt):
py -m venv .venv
. .venv\Scripts\activate
pip install -r requirements-win.txt
```

### Environment Variables

Enter the backend folder
```bash
cd backend
```

Create a file named `.env` (no local this time) at the top of the backend folder, and place the following line within.
```bash
FRONTEND_URL=http://127.0.0.1:3000
```

### Running the server
Run the server with Flask while within the backend folder (so that `app.py` is visible), which should default the port to 5000. Again, you can use Ctrl-C to exit.
```bash
flask --app app run
```

After this step, the frontend should have full functionality.


