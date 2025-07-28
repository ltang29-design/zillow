#!/usr/bin/env python3
"""
Command Line Interface for Housing Market Prediction System.
Provides commands for data ingestion, database management, and system administration.
"""

import typer
import asyncio
from pathlib import Path
from typing import Optional
import logging
from rich.console import Console
from rich.table import Table
from rich.progress import track
from rich import print as rprint

# Import our modules
from app.core.config import settings
from app.core.database import database, init_database, reset_database, check_database_health
from app.data_ingestion.zillow_transformer import ZillowDataTransformer, run_data_ingestion

# Initialize CLI app and console
app = typer.Typer(help="Housing Market Prediction System CLI")
console = Console()

# Configure logging
def setup_logging(verbose: bool = False):
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


@app.command()
def init_db(
    reset: bool = typer.Option(False, "--reset", help="Reset database (WARNING: This will delete all data)"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose logging")
):
    """Initialize the database with required tables."""
    setup_logging(verbose)
    
    async def _init():
        if reset:
            if typer.confirm("⚠️  This will delete ALL data. Are you sure?"):
                console.print("🗑️  Resetting database...", style="bold red")
                await reset_database()
                console.print("✅ Database reset complete", style="bold green")
            else:
                console.print("❌ Database reset cancelled", style="yellow")
                return
        else:
            console.print("🚀 Initializing database...", style="bold blue")
            await init_database()
            console.print("✅ Database initialized successfully", style="bold green")
    
    asyncio.run(_init())


@app.command()
def health_check(
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose logging")
):
    """Check system health including database connectivity."""
    setup_logging(verbose)
    
    async def _health_check():
        console.print("🔍 Checking system health...", style="bold blue")
        
        # Database health
        db_healthy = await check_database_health()
        
        # Create health table
        table = Table(title="System Health Check")
        table.add_column("Component", style="cyan")
        table.add_column("Status", style="bold")
        table.add_column("Details")
        
        # Database status
        db_status = "✅ Healthy" if db_healthy else "❌ Unhealthy"
        db_style = "green" if db_healthy else "red"
        table.add_row("Database", f"[{db_style}]{db_status}[/{db_style}]", settings.database_url.split('@')[1] if '@' in settings.database_url else settings.database_url)
        
        # Configuration status
        config_issues = []
        if not settings.fred_api_key or settings.fred_api_key == "your_fred_api_key_here":
            config_issues.append("FRED API key not set")
        
        config_status = "✅ OK" if not config_issues else f"⚠️  Issues: {', '.join(config_issues)}"
        config_style = "green" if not config_issues else "yellow"
        table.add_row("Configuration", f"[{config_style}]{config_status}[/{config_style}]", f"Environment: {settings.environment}")
        
        console.print(table)
        
        if not db_healthy:
            console.print("❌ System is not healthy. Check database connection.", style="bold red")
        elif config_issues:
            console.print("⚠️  System is partially healthy. Check configuration.", style="bold yellow")
        else:
            console.print("✅ System is healthy and ready!", style="bold green")
    
    asyncio.run(_health_check())


@app.command()
def ingest_data(
    data_path: str = typer.Option("./data", "--data-path", "-d", help="Path to data directory"),
    file_pattern: str = typer.Option("Metro_*.csv", "--pattern", "-p", help="File pattern to match"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose logging")
):
    """Ingest Zillow CSV data into the database."""
    setup_logging(verbose)
    
    console.print(f"🔄 Starting data ingestion from {data_path}...", style="bold blue")
    
    # Check if data path exists
    data_dir = Path(data_path)
    if not data_dir.exists():
        console.print(f"❌ Data directory not found: {data_path}", style="bold red")
        raise typer.Exit(1)
    
    # Find CSV files
    csv_files = list(data_dir.glob(file_pattern))
    if not csv_files:
        console.print(f"❌ No CSV files found matching pattern: {file_pattern}", style="bold red")
        raise typer.Exit(1)
    
    console.print(f"📁 Found {len(csv_files)} CSV files", style="bold cyan")
    
    # Run ingestion
    try:
        results = run_data_ingestion(data_path)
        
        # Display results table
        table = Table(title="Data Ingestion Results")
        table.add_column("File", style="cyan")
        table.add_column("Status", style="bold")
        
        for filename, success in results.items():
            status = "✅ Success" if success else "❌ Failed"
            style = "green" if success else "red"
            table.add_row(filename, f"[{style}]{status}[/{style}]")
        
        console.print(table)
        
        # Summary
        successful = sum(results.values())
        total = len(results)
        if successful == total:
            console.print(f"✅ All {total} files processed successfully!", style="bold green")
        else:
            console.print(f"⚠️  {successful}/{total} files processed successfully", style="bold yellow")
            
    except Exception as e:
        console.print(f"❌ Data ingestion failed: {e}", style="bold red")
        raise typer.Exit(1)


@app.command()
def data_summary(
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose logging")
):
    """Display summary of data in the database."""
    setup_logging(verbose)
    
    from app.core.database import get_sync_session
    from app.models.database import Metro, HousingMetric
    from sqlalchemy import func
    
    console.print("📊 Generating data summary...", style="bold blue")
    
    try:
        with next(get_sync_session()) as session:
            transformer = ZillowDataTransformer()
            summary = transformer.get_data_summary(session)
            
            # Create summary table
            table = Table(title="Database Summary")
            table.add_column("Metric", style="cyan")
            table.add_column("Value", style="bold")
            table.add_column("Details")
            
            # Add metrics
            table.add_row("Metro Areas", str(summary['metro_count']), "Unique metropolitan areas")
            table.add_row("Housing Records", str(summary['housing_metrics_count']), "Total housing metric records")
            
            if summary['date_range']['start'] and summary['date_range']['end']:
                date_range = f"{summary['date_range']['start'].strftime('%Y-%m-%d')} to {summary['date_range']['end'].strftime('%Y-%m-%d')}"
                table.add_row("Date Range", date_range, "Temporal coverage")
            
            target_pct = summary['target_variable_coverage']['percentage']
            target_status = f"{target_pct:.1f}%"
            target_style = "green" if target_pct > 80 else "yellow" if target_pct > 50 else "red"
            table.add_row("Target Coverage", f"[{target_style}]{target_status}[/{target_style}]", "Median days to pending data availability")
            
            console.print(table)
            
            # Additional metrics
            if summary['housing_metrics_count'] > 0:
                console.print(f"✅ Database contains {summary['housing_metrics_count']:,} housing records across {summary['metro_count']} metros", style="bold green")
            else:
                console.print("⚠️  Database appears to be empty. Run 'ingest-data' first.", style="bold yellow")
                
    except Exception as e:
        console.print(f"❌ Failed to generate summary: {e}", style="bold red")
        raise typer.Exit(1)


@app.command()
def list_files(
    data_path: str = typer.Option("./data", "--data-path", "-d", help="Path to data directory"),
    pattern: str = typer.Option("*.csv", "--pattern", "-p", help="File pattern to match")
):
    """List available data files."""
    console.print(f"📂 Listing files in {data_path}...", style="bold blue")
    
    data_dir = Path(data_path)
    if not data_dir.exists():
        console.print(f"❌ Directory not found: {data_path}", style="bold red")
        raise typer.Exit(1)
    
    files = list(data_dir.glob(pattern))
    
    if not files:
        console.print(f"📭 No files found matching pattern: {pattern}", style="yellow")
        return
    
    # Create file table
    table = Table(title=f"Files in {data_path}")
    table.add_column("Filename", style="cyan")
    table.add_column("Size", style="magenta")
    table.add_column("Modified", style="green")
    table.add_column("Recognized", style="bold")
    
    transformer = ZillowDataTransformer()
    
    for file_path in sorted(files):
        stat = file_path.stat()
        size = f"{stat.st_size / 1024:.1f} KB"
        modified = f"{stat.st_mtime}"
        
        # Check if file type is recognized
        file_mapping = transformer.identify_file_type(file_path.name)
        recognized = "✅ Yes" if file_mapping else "❌ No"
        recognized_style = "green" if file_mapping else "red"
        
        table.add_row(
            file_path.name,
            size,
            modified,
            f"[{recognized_style}]{recognized}[/{recognized_style}]"
        )
    
    console.print(table)
    console.print(f"📊 Found {len(files)} files", style="bold blue")


@app.command()
def config_info():
    """Display current configuration."""
    table = Table(title="Configuration")
    table.add_column("Setting", style="cyan")
    table.add_column("Value", style="bold")
    table.add_column("Source")
    
    # Safe settings to display (no secrets)
    safe_settings = [
        ("Environment", settings.environment, "env"),
        ("Debug Mode", str(settings.debug), "env"),
        ("Database Host", settings.database_host, "env"),
        ("Database Port", str(settings.database_port), "env"),
        ("Database Name", settings.database_name, "env"),
        ("Redis Host", settings.redis_host, "env"),
        ("Data Path", settings.zillow_data_path, "env"),
        ("Model Refresh Hours", str(settings.model_refresh_hours), "env"),
        ("Log Level", settings.log_level, "env"),
    ]
    
    for setting, value, source in safe_settings:
        table.add_row(setting, value, source)
    
    console.print(table)


@app.command()
def fetch_external_data(
    source: str = typer.Option("all", "--source", "-s", help="Data source: fred, census, or all"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose logging")
):
    """Fetch external economic and demographic data."""
    setup_logging(verbose)
    
    console.print(f"🌐 Fetching external data from: {source}", style="bold blue")
    
    async def _fetch_data():
        results = {}
        
        if source in ["fred", "all"]:
            console.print("📊 Fetching FRED economic data...", style="cyan")
            try:
                from app.external_data.fred_client import run_fred_update
                fred_summary = await run_fred_update()
                results["fred"] = fred_summary
                
                if fred_summary.get('errors'):
                    console.print(f"⚠️  FRED fetch completed with errors: {fred_summary['errors']}", style="yellow")
                else:
                    console.print(f"✅ FRED data fetched: {fred_summary.get('records_saved', 0)} records", style="green")
                    
            except Exception as e:
                error_msg = f"❌ FRED fetch failed: {e}"
                console.print(error_msg, style="red")
                results["fred"] = {"error": str(e)}
        
        if source in ["census", "all"]:
            console.print("🏛️  Fetching Census demographic data...", style="cyan")
            try:
                from app.external_data.census_client import run_census_update
                census_summary = await run_census_update()
                results["census"] = census_summary
                
                if census_summary.get('errors'):
                    console.print(f"⚠️  Census fetch completed with errors: {census_summary['errors']}", style="yellow")
                else:
                    console.print(f"✅ Census data fetched: {census_summary.get('total_records', 0)} records", style="green")
                    
            except Exception as e:
                error_msg = f"❌ Census fetch failed: {e}"
                console.print(error_msg, style="red")
                results["census"] = {"error": str(e)}
        
        # Display summary table
        table = Table(title="External Data Fetch Results")
        table.add_column("Source", style="cyan")
        table.add_column("Status", style="bold")
        table.add_column("Records", style="magenta")
        table.add_column("Duration")
        
        for source_name, summary in results.items():
            if "error" in summary:
                status = "❌ Failed"
                records = "0"
                duration = "N/A"
            else:
                status = "✅ Success"
                records = str(summary.get('records_saved', summary.get('total_records', 0)))
                duration = str(summary.get('duration', 'N/A'))
            
            table.add_row(source_name.upper(), status, records, duration)
        
        console.print(table)
        return results
    
    results = asyncio.run(_fetch_data())
    
    # Summary
    success_count = sum(1 for r in results.values() if "error" not in r)
    total_count = len(results)
    
    if success_count == total_count:
        console.print(f"🎉 All {total_count} data sources fetched successfully!", style="bold green")
    else:
        console.print(f"⚠️  {success_count}/{total_count} data sources completed successfully", style="bold yellow")


@app.command()
def test_api(
    base_url: str = typer.Option("http://localhost:8000", "--url", help="Base URL for API testing"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose logging")
):
    """Test the API endpoints."""
    setup_logging(verbose)
    
    import requests
    
    console.print(f"🧪 Testing API endpoints at {base_url}...", style="bold blue")
    
    tests = [
        ("Health Check", f"{base_url}/health"),
        ("Detailed Health", f"{base_url}/health/detailed"),
        ("Root Endpoint", f"{base_url}/"),
        ("Data Summary", f"{base_url}/api/v1/data/summary"),
        ("Metro List", f"{base_url}/api/v1/metros?limit=5"),
    ]
    
    results = []
    
    for test_name, url in tests:
        try:
            response = requests.get(url, timeout=10)
            status = "✅ Pass" if response.status_code == 200 else f"❌ Fail ({response.status_code})"
            results.append((test_name, status, response.status_code, response.elapsed.total_seconds()))
            
            if verbose and response.status_code == 200:
                console.print(f"📝 {test_name} response: {response.json()}")
                
        except requests.exceptions.RequestException as e:
            results.append((test_name, f"❌ Error", "N/A", "N/A"))
            if verbose:
                console.print(f"Error testing {test_name}: {e}")
    
    # Display results table
    table = Table(title="API Test Results")
    table.add_column("Test", style="cyan")
    table.add_column("Status", style="bold")
    table.add_column("Status Code")
    table.add_column("Response Time")
    
    for test_name, status, status_code, response_time in results:
        table.add_row(
            test_name, 
            status, 
            str(status_code), 
            f"{response_time:.3f}s" if isinstance(response_time, (int, float)) else str(response_time)
        )
    
    console.print(table)
    
    # Summary
    passed = sum(1 for _, status, _, _ in results if "Pass" in status)
    total = len(results)
    
    if passed == total:
        console.print(f"✅ All {total} API tests passed!", style="bold green")
    else:
        console.print(f"⚠️  {passed}/{total} API tests passed", style="bold yellow")


@app.command()
def start_api(
    host: str = typer.Option("0.0.0.0", "--host", help="Host to bind to"),
    port: int = typer.Option(8000, "--port", help="Port to bind to"),
    reload: bool = typer.Option(False, "--reload", help="Enable auto-reload for development"),
    workers: int = typer.Option(1, "--workers", help="Number of worker processes")
):
    """Start the FastAPI server."""
    console.print(f"🚀 Starting API server on {host}:{port}...", style="bold blue")
    
    if reload and workers > 1:
        console.print("⚠️  Cannot use --reload with multiple workers. Setting workers=1", style="yellow")
        workers = 1
    
    import uvicorn
    
    try:
        uvicorn.run(
            "app.main:app",
            host=host,
            port=port,
            reload=reload,
            workers=workers,
            log_level=settings.log_level.lower()
        )
    except KeyboardInterrupt:
        console.print("\n👋 Server stopped", style="bold yellow")


if __name__ == "__main__":
    app() 