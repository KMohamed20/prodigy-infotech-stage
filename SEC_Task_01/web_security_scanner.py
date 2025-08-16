"""
SEC Task 01 - Web Security Scanner
Prodigy InfoTech Internship | Khalid Ag Mohamed Aly
August 2025

Objective: Analyze a website for common security vulnerabilities.
"""

import requests
import ssl
import socket
from urllib.parse import urlparse
import datetime

# Disable SSL warnings for insecure sites
requests.packages.urllib3.disable_warnings()

def check_security_headers(url):
    """Check for essential security headers"""
    try:
        response = requests.get(url, timeout=10, verify=False)
        headers = response.headers

        security_headers = {
            'Strict-Transport-Security': headers.get('Strict-Transport-Security', 'Missing'),
            'X-Content-Type-Options': headers.get('X-Content-Type-Options', 'Missing'),
            'X-Frame-Options': headers.get('X-Frame-Options', 'Missing'),
            'Content-Security-Policy': headers.get('Content-Security-Policy', 'Missing'),
            'X-XSS-Protection': headers.get('X-XSS-Protection', 'Missing'),
            'Referrer-Policy': headers.get('Referrer-Policy', 'Missing')
        }

        return security_headers
    except Exception as e:
        print(f"Error checking headers: {e}")
        return {}

def check_ssl_certificate(url):
    """Check SSL/TLS certificate validity"""
    try:
        parsed = urlparse(url)
        hostname = parsed.netloc.split(':')[0] if ':' in parsed.netloc else parsed.netloc

        context = ssl.create_default_context()
        with socket.create_connection((hostname, 443), timeout=10) as sock:
            with context.wrap_socket(sock, server_hostname=hostname) as ssock:
                cert = ssock.getpeercert()
                return {
                    'valid': True,
                    'subject': dict(x[0] for x in cert['subject']),
                    'issuer': dict(x[0] for x in cert['issuer']),
                    'expiry': cert['notAfter']
                }
    except Exception as e:
        return {'valid': False, 'error': str(e)}

def check_basic_vulnerabilities(url):
    """Check for basic vulnerabilities"""
    try:
        response = requests.get(url, timeout=10)
        content = response.text.lower()

        vulnerabilities = {
            'SQLi': 'sql syntax' in content or 'mysql error' in content,
            'XSS': '<script>' in content or 'alert(' in content,
            'Directory_Listing': 'index of /' in content or 'directory listing' in content
        }

        return vulnerabilities
    except Exception as e:
        return {'error': str(e)}

def generate_report(url):
    """Generate comprehensive security report"""
    print(f"\n🔍 Security Report for: {url}")
    print(f"Generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)

    # Check headers
    headers = check_security_headers(url)
    print("\n🛡️ SECURITY HEADERS:")
    for header, value in headers.items():
        status = "✅" if value != 'Missing' else "❌"
        print(f"  {status} {header}: {value}")

    # Check SSL
    ssl_info = check_ssl_certificate(url)
    print("\n🔐 SSL/TLS CERTIFICATE:")
    if ssl_info['valid']:
        print("  ✅ Certificate is valid")
        print(f"     Expires: {ssl_info['expiry']}")
    else:
        print(f"  ❌ Invalid or missing certificate: {ssl_info.get('error', 'Unknown')}")

    # Check vulnerabilities
    vulns = check_basic_vulnerabilities(url)
    print("\n⚠️ BASIC VULNERABILITIES:")
    for vuln, found in vulns.items():
        if isinstance(found, bool):
            status = "✅ No" if not found else "❌ YES"
            print(f"  {status} {vuln}")
        else:
            print(f"  ⚠️  {vuln}: Error - {found}")

    # Risk summary
    missing_headers = sum(1 for v in headers.values() if v == 'Missing')
    risk_level = "Low" if missing_headers <= 2 and ssl_info['valid'] and not any(vulns.values()) else "High"
    print(f"\n📊 RISK LEVEL: {'🟢 ' + risk_level if risk_level == 'Low' else '🔴 ' + risk_level}")

if __name__ == "__main__":
    target_url = input("Enter website URL to scan (e.g., https://example.com): ").strip()
    if not target_url.startswith('http'):
        target_url = 'https://' + target_url

    generate_report(target_url)
