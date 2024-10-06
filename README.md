# Stakeholder Mapping Tool

This tool is a website built on Next.js and React, intended to identify stakeholders from user-input documents and generate a graph of coalitions between stakeholders.

## Testing in backend folder
Create a virtual environment to contain Python dependencies, if you'd like. If you use Python's `venv` module, then the example below will create a new virtual environment with name ".venv".
```bash
# MacOS
python3 -m venv .venv

# Windows
py -m venv .venv
```

With a virtual environment created this way, enter it:
```bash
# Mac
source .venv/bin/activate

# Windows CMD
.venv\Scripts\activate
```

Your terminal prompt should then have "(.venv)" in front to indicate that you are within the virtual env. To install packages:
```bash
pip install -r requirements.txt
```

To leave the virtual environment, either close the terminal or run:
```bash
deactivate
```

## Running the Website Locally
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

Open [http://localhost:3000](http://localhost:3000) with your browser to see the result. Making any changes in code and saving the file should update the localhost instance automatically.
