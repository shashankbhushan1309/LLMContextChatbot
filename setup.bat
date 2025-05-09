@echo off
echo Setting up PDF Question Answering System...

echo Creating virtual environment...
python -m venv venv

echo Activating virtual environment...
call venv\Scripts\activate.bat

echo Installing dependencies using setup script...
python setup.py

echo.
echo If setup completed successfully, you can now:
echo 1. Edit config.env to add your Gemini API key
echo 2. Run 'python app.py' to start the application
echo.
echo Press any key to exit...
pause >nul