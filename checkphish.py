import sys
print(f"Running with Python executable: {sys.executable}")
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse, urljoin
import re
from typing import Dict, List, Tuple
import ssl
import socket
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os
import sys
import io

# Windows UTF-8 fix - MUST BE EARLY
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

class PhishingDetector:
    def __init__(self, use_ml=True, ml_only=False):
        self.use_ml = use_ml
        self.ml_only = ml_only
        self.model = None
        self.model_path = 'phishing_model.pkl'
        
        # Load model if it exists
        if use_ml and os.path.exists(self.model_path):
            self.model = joblib.load(self.model_path)
            print(f"✓ Loaded trained model from {self.model_path}")
        
    def extract_features(self, url: str, soup=None, results=None) -> np.ndarray:
        """Extract 30+ features for ML model"""
        features = []
        
        try:
            parsed = urlparse(url)
            domain = parsed.netloc.lower()
            path = parsed.path.lower()
            
            # URL-based features (15 features)
            features.append(1 if parsed.scheme == 'https' else 0)  # Has HTTPS
            features.append(len(url))  # URL length
            features.append(domain.count('.'))  # Number of dots
            features.append(domain.count('-'))  # Number of hyphens
            features.append(domain.count('_'))  # Number of underscores
            features.append(1 if re.search(r'\d', domain) else 0)  # Has digits in domain
            features.append(len(domain))  # Domain length
            features.append(len(path))  # Path length
            features.append(1 if '@' in url else 0)  # Has @ symbol
            features.append(1 if '//' in path else 0)  # Double slash in path
            features.append(url.count('?'))  # Number of query parameters
            features.append(url.count('&'))  # Number of ampersands
            features.append(1 if re.match(r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', domain) else 0)  # Is IP
            
            # Check suspicious TLDs
            suspicious_tlds = ['.tk', '.ml', '.ga', '.cf', '.gq', '.xyz', '.top', '.work']
            features.append(1 if any(domain.endswith(tld) for tld in suspicious_tlds) else 0)
            
            # Check for unicode/IDN
            features.append(1 if not all(ord(c) < 128 for c in domain) else 0)
            
            # Content-based features (15 features)
            if soup:
                # Form features
                forms = soup.find_all('form')
                features.append(len(forms))  # Number of forms
                
                password_fields = soup.find_all('input', {'type': 'password'})
                features.append(len(password_fields))  # Number of password fields
                
                # Check for forms with validation
                forms_with_validation = 0
                for form in forms:
                    inputs = form.find_all(['input', 'textarea'])
                    if any(inp.get('required') or inp.get('pattern') for inp in inputs):
                        forms_with_validation += 1
                features.append(forms_with_validation)  # Forms with validation
                
                # External links
                links = soup.find_all('a', href=True)
                external_links = sum(1 for link in links if urlparse(link['href']).netloc and urlparse(link['href']).netloc != domain)
                features.append(external_links)  # External links count
                
                # Images
                images = soup.find_all('img')
                external_images = sum(1 for img in images if img.get('src') and urlparse(img.get('src', '')).netloc and urlparse(img.get('src', '')).netloc != domain)
                features.append(len(images))  # Total images
                features.append(external_images)  # External images
                
                # Scripts
                scripts = soup.find_all('script')
                external_scripts = sum(1 for script in scripts if script.get('src') and urlparse(script.get('src', '')).netloc and urlparse(script.get('src', '')).netloc != domain)
                features.append(len(scripts))  # Total scripts
                features.append(external_scripts)  # External scripts
                
                # Hidden elements
                hidden_elements = len(soup.find_all(style=re.compile(r'display\s*:\s*none|visibility\s*:\s*hidden')))
                features.append(hidden_elements)  # Hidden elements
                
                # IFrames
                iframes = soup.find_all('iframe')
                features.append(len(iframes))  # Number of iframes
                
                # Text analysis
                text = soup.get_text().lower()
                urgency_keywords = ['urgent', 'suspended', 'verify', 'confirm', 'update', 'expire', 'limited', 'act now', 'click here', 'winner']
                urgency_count = sum(1 for keyword in urgency_keywords if keyword in text)
                features.append(urgency_count)  # Urgency keywords
                
                # Meta tags
                meta_tags = soup.find_all('meta')
                features.append(len(meta_tags))  # Number of meta tags
                
                # Title length
                title = soup.find('title')
                features.append(len(title.string) if title and title.string else 0)  # Title length
                
                # Favicon
                favicon = soup.find('link', rel=re.compile('icon', re.I))
                features.append(1 if favicon else 0)  # Has favicon
                
                # SSL seal/security badges (common in phishing)
                ssl_text = ['ssl', 'secure', 'verified', 'trusted', 'security']
                ssl_badge_count = sum(1 for keyword in ssl_text if keyword in text)
                features.append(min(ssl_badge_count, 5))  # SSL badge mentions (capped at 5)
                
            else:
                # If no soup, fill with zeros
                features.extend([0] * 15)
                
        except Exception as e:
            print(f"Error extracting features: {e}")
            # Return zeros if feature extraction fails
            features.extend([0] * (30 - len(features)))
        
        # Ensure we have exactly 30 features
        while len(features) < 30:
            features.append(0)
        
        return np.array(features[:30])
    
    def train_model(self, training_data: List[Tuple[str, int]], model_type='random_forest'):
        """Train ML model on labeled data
        
        Args:
            training_data: List of tuples (url, label) where label is 0 (legitimate) or 1 (phishing)
            model_type: Type of model to train ('random_forest', 'svm', etc.)
        """
        print(f"\n🎓 Training {model_type} model on {len(training_data)} samples...")
        
        X = []
        y = []
        
        for url, label in training_data:
            try:
                # Try to fetch and parse the page
                response = requests.get(url, timeout=10, headers={
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                })
                soup = BeautifulSoup(response.content, 'html.parser')
                features = self.extract_features(url, soup)
                X.append(features)
                y.append(label)
            except Exception as e:
                print(f"⚠️  Skipping {url}: {e}")
                continue
        
        X = np.array(X)
        y = np.array(y)
        
        print(f"✓ Successfully extracted features from {len(X)} websites")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train model
        if model_type == 'random_forest':
            self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        
        self.model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"\n📊 Model Performance:")
        print(f"Accuracy: {accuracy:.2%}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=['Legitimate', 'Phishing']))
        
        # Save model
        joblib.dump(self.model, self.model_path)
        print(f"\n✓ Model saved to {self.model_path}")
        
        return accuracy
    
    def analyze_url(self, url: str) -> Dict:
        """Main function to analyze a URL for phishing indicators"""
        if not url.startswith(('http://', 'https://')):
            url = 'https://' + url
            
        results = {
            'url': url,
            'is_suspicious': False,
            'risk_level': 'Unknown',
            'findings': [],
            'forms_analyzed': 0,
            'sensitive_forms': 0,
            'has_validation': False,
            'ssl_valid': False,
            'ml_prediction': None,
            'ml_confidence': None
        }
        
        try:
            # ML-only mode
            if self.ml_only:
                if not self.use_ml or not self.model:
                    results['findings'].append("❌ ML model not loaded. Cannot perform ML-only analysis.")
                    results['risk_level'] = 'Error'
                    return results
                
                # Fetch webpage for feature extraction
                response = requests.get(url, timeout=10, headers={
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                })
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Extract features and predict
                features = self.extract_features(url, soup, results)
                prediction = self.model.predict([features])[0]
                confidence = self.model.predict_proba([features])[0]
                
                # Get legitimate and phishing confidence
                legitimate_confidence = float(confidence[0]) * 100  # Class 0 = Legitimate
                phishing_confidence = float(confidence[1]) * 100     # Class 1 = Phishing
                
                # Custom threshold: Legitimate only if confidence > 51%
                if legitimate_confidence > 51:
                    results['ml_prediction'] = 'Legitimate'
                    results['ml_confidence'] = legitimate_confidence
                else:
                    results['ml_prediction'] = 'Phishing'
                    results['ml_confidence'] = phishing_confidence if phishing_confidence > legitimate_confidence else (100 - legitimate_confidence)
                
                # Set risk level based on ML prediction only
                if results['ml_prediction'] == 'Phishing':
                    if results['ml_confidence'] > 90:
                        results['risk_level'] = 'Critical'
                    elif results['ml_confidence'] > 80:
                        results['risk_level'] = 'High'
                    elif results['ml_confidence'] > 70:
                        results['risk_level'] = 'Medium'
                    else:
                        results['risk_level'] = 'Low'
                    results['is_suspicious'] = True
                    results['findings'].append(f"🤖 ML Model predicts: PHISHING (confidence: {results['ml_confidence']:.1f}%)")
                else:
                    results['risk_level'] = 'Safe'
                    results['is_suspicious'] = False
                    results['findings'].append(f"🤖 ML Model predicts: LEGITIMATE (confidence: {results['ml_confidence']:.1f}%)")
                
                return results
            
            # Standard mode (with all checks)
            # Check SSL certificate
            results['ssl_valid'] = self.check_ssl(url)
            if not results['ssl_valid']:
                results['findings'].append("⚠️ No valid SSL certificate (HTTPS)")
            
            # Fetch webpage
            response = requests.get(url, timeout=10, headers={
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            })
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # ML Prediction
            if self.use_ml and self.model:
                features = self.extract_features(url, soup, results)
                prediction = self.model.predict([features])[0]
                confidence = self.model.predict_proba([features])[0]
                
                # Get legitimate and phishing confidence
                legitimate_confidence = float(confidence[0]) * 100  # Class 0 = Legitimate
                phishing_confidence = float(confidence[1]) * 100     # Class 1 = Phishing
                
                # Custom threshold: Legitimate only if confidence > 51%
                if legitimate_confidence > 51:
                    results['ml_prediction'] = 'Legitimate'
                    results['ml_confidence'] = legitimate_confidence
                    results['findings'].append(f"🤖 ML Model predicts: LEGITIMATE (confidence: {results['ml_confidence']:.1f}%)")
                else:
                    results['ml_prediction'] = 'Phishing'
                    results['ml_confidence'] = phishing_confidence if phishing_confidence > legitimate_confidence else (100 - legitimate_confidence)
                    results['findings'].append(f"🤖 ML Model predicts: PHISHING (confidence: {results['ml_confidence']:.1f}%)")

            
            # Analyze forms
            forms = soup.find_all('form')
            results['forms_analyzed'] = len(forms)
            
            sensitive_forms = []
            for form in forms:
                form_analysis = self.analyze_form(form, url)
                if form_analysis['has_sensitive_fields']:
                    sensitive_forms.append(form_analysis)
                    results['sensitive_forms'] += 1
            
            # Check for validation in sensitive forms
            if sensitive_forms:
                validation_count = sum(1 for f in sensitive_forms if f['has_validation'])
                results['has_validation'] = validation_count > 0
                
                for form_data in sensitive_forms:
                    if form_data['has_sensitive_fields']:
                        field_types = ', '.join(form_data['sensitive_field_types'])
                        if form_data['has_validation']:
                            results['findings'].append(
                                f"✓ Form with {field_types} has validation (Good sign)"
                            )
                        else:
                            results['findings'].append(
                                f"⚠️ Form with {field_types} lacks validation (Suspicious)"
                            )
            
            # Additional phishing indicators
            self.check_url_patterns(url, results)
            self.check_page_content(soup, results)
            
            # Apply custom logic for full mode (combining ML + form validation)
            if self.use_ml and self.model:
                # Condition 4: ML says legitimate, but sensitive forms WITHOUT validation
                # Only override if ML confidence is <= 70%
                if results['ml_prediction'] == 'Legitimate' and results['sensitive_forms'] > 0 and not results['has_validation']:
                    if results['ml_confidence'] > 70:
                        # High confidence legitimate - trust ML, ignore form validation
                        results['is_suspicious'] = False
                        results['risk_level'] = 'Safe'
                        results['findings'].append("✅ High ML confidence (>70%) - trusting ML prediction despite missing validation")
                    else:
                        # Low confidence legitimate + no validation = PHISHING
                        results['is_suspicious'] = True
                        results['risk_level'] = 'High'
                        results['findings'].append("🚨 Override: ML says legitimate but sensitive forms lack validation - PHISHING")
                
                # Condition 1: ML says phishing, but no sensitive forms -> NOT phishing
                elif results['ml_prediction'] == 'Phishing' and results['sensitive_forms'] == 0:
                    results['is_suspicious'] = False
                    results['risk_level'] = 'Safe'
                    results['findings'].append("✅ Override: No sensitive forms found, likely a false positive")
                
                # Condition 2: ML says phishing, and sensitive forms WITHOUT validation -> PHISHING
                elif results['ml_prediction'] == 'Phishing' and results['sensitive_forms'] > 0 and not results['has_validation']:
                    results['is_suspicious'] = True
                    results['risk_level'] = 'Critical'
                    results['findings'].append("🚨 Confirmed: Sensitive forms without validation detected")
                
                # Condition 3: ML says phishing, and sensitive forms WITH validation -> Trust ML prediction (PHISHING)
                elif results['ml_prediction'] == 'Phishing' and results['sensitive_forms'] > 0 and results['has_validation']:
                    results['is_suspicious'] = True  # Trust ML: it says Phishing
                    results['findings'].append("⚠️ Forms have validation, trusting ML prediction: PHISHING")
                    # Set risk level based on ML confidence
                    if results['ml_confidence'] > 90:
                        results['risk_level'] = 'Critical'
                    elif results['ml_confidence'] > 70:
                        results['risk_level'] = 'High'
                    elif results['ml_confidence'] > 50:
                        results['risk_level'] = 'Medium'
                    else:
                        results['risk_level'] = 'Low'
                
                # ML says legitimate and either no sensitive forms or forms have validation
                else:
                    # Calculate risk level normally
                    results['risk_level'] = self.calculate_risk_level(results)
                    results['is_suspicious'] = results['risk_level'] in ['High', 'Critical']
            else:
                # ML not available: Calculate risk level normally
                results['risk_level'] = self.calculate_risk_level(results)
                results['is_suspicious'] = results['risk_level'] in ['High', 'Critical']
            
        except requests.RequestException as e:
            results['findings'].append(f"❌ Error fetching URL: {str(e)}")
            results['risk_level'] = 'Error'
        
        return results
    
    def analyze_form(self, form, base_url: str) -> Dict:
        """Analyze a single form for sensitive fields and validation"""
        analysis = {
            'has_sensitive_fields': False,
            'sensitive_field_types': [],
            'has_validation': False,
            'validation_types': [],
            'action': form.get('action', ''),
            'method': form.get('method', 'get').lower()
        }
        
        # Get all input fields
        inputs = form.find_all(['input', 'textarea'])
        
        sensitive_patterns = {
            'password': ['password', 'passwd', 'pwd'],
            'email': ['email', 'e-mail'],
            'credit_card': ['card', 'cardnumber', 'cc', 'creditcard'],
            'ssn': ['ssn', 'social'],
            'cvv': ['cvv', 'cvc', 'security-code']
        }
        
        for inp in inputs:
            input_type = inp.get('type', 'text').lower()
            input_name = inp.get('name', '').lower()
            input_id = inp.get('id', '').lower()
            
            # Check if field is sensitive
            for field_type, patterns in sensitive_patterns.items():
                if (input_type in patterns or 
                    any(p in input_name for p in patterns) or 
                    any(p in input_id for p in patterns)):
                    analysis['has_sensitive_fields'] = True
                    if field_type not in analysis['sensitive_field_types']:
                        analysis['sensitive_field_types'].append(field_type)
            
            # Check for validation attributes
            if inp.get('required') is not None:
                analysis['has_validation'] = True
                analysis['validation_types'].append('required')
            
            if inp.get('pattern'):
                analysis['has_validation'] = True
                analysis['validation_types'].append('pattern')
            
            if inp.get('minlength') or inp.get('maxlength'):
                analysis['has_validation'] = True
                analysis['validation_types'].append('length')
            
            if input_type in ['email', 'url', 'tel', 'number']:
                analysis['has_validation'] = True
                analysis['validation_types'].append('type-validation')
        
        # Check for JavaScript validation
        scripts = form.find_all('script')
        for script in scripts:
            script_text = script.string or ''
            if any(keyword in script_text for keyword in ['validate', 'validation', 'required', 'checkForm']):
                analysis['has_validation'] = True
                analysis['validation_types'].append('javascript')
        
        return analysis
    
    def check_ssl(self, url: str) -> bool:
        """Check if URL has valid SSL certificate"""
        try:
            parsed = urlparse(url)
            if parsed.scheme != 'https':
                return False
            
            hostname = parsed.netloc
            context = ssl.create_default_context()
            with socket.create_connection((hostname, 443), timeout=5) as sock:
                with context.wrap_socket(sock, server_hostname=hostname) as ssock:
                    return True
        except:
            return False
    
    def check_url_patterns(self, url: str, results: Dict):
        """Check URL for suspicious patterns"""
        parsed = urlparse(url)
        domain = parsed.netloc.lower()
        
        # Check for IP address instead of domain
        ip_pattern = r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}'
        if re.match(ip_pattern, domain):
            results['findings'].append("⚠️ URL uses IP address instead of domain name")
        
        # Check for suspicious TLDs
        suspicious_tlds = ['.tk', '.ml', '.ga', '.cf', '.gq', '.xyz']
        if any(domain.endswith(tld) for tld in suspicious_tlds):
            results['findings'].append("⚠️ Suspicious top-level domain (TLD)")
        
        # Check for excessive subdomains or hyphens
        if domain.count('.') > 3 or domain.count('-') > 2:
            results['findings'].append("⚠️ Suspicious domain structure (too many dots/hyphens)")
        
        # Check for homograph attacks (unicode lookalikes)
        if not all(ord(c) < 128 for c in domain):
            results['findings'].append("⚠️ Contains non-ASCII characters (possible homograph attack)")
    
    def check_page_content(self, soup: BeautifulSoup, results: Dict):
        """Check page content for phishing indicators"""
        # Check for urgency keywords
        urgency_keywords = ['urgent', 'suspended', 'verify', 'confirm', 'update', 'expire', 'limited time']
        text = soup.get_text().lower()
        
        urgency_found = [kw for kw in urgency_keywords if kw in text]
        if len(urgency_found) >= 3:
            results['findings'].append(f"⚠️ Multiple urgency keywords detected: {', '.join(urgency_found[:3])}")
        
        # Check for hidden iframes
        iframes = soup.find_all('iframe')
        hidden_iframes = [iframe for iframe in iframes if 
                         iframe.get('style', '').find('display:none') != -1 or
                         iframe.get('style', '').find('visibility:hidden') != -1]
        if hidden_iframes:
            results['findings'].append(f"⚠️ {len(hidden_iframes)} hidden iframe(s) detected")
    
    def calculate_risk_level(self, results: Dict) -> str:
        """Calculate overall risk level"""
        risk_score = 0
        
        # ML prediction weight (highest priority)
        if results.get('ml_prediction') == 'Phishing':
            risk_score += 50 if results.get('ml_confidence', 0) > 80 else 30
        
        # No SSL = major red flag
        if not results['ssl_valid']:
            risk_score += 20
        
        # Sensitive forms without validation
        if results['sensitive_forms'] > 0 and not results['has_validation']:
            risk_score += 30
        
        # Count warning indicators
        warning_count = len([f for f in results['findings'] if '⚠️' in f])
        risk_score += warning_count * 5
        
        if risk_score >= 70:
            return 'Critical'
        elif risk_score >= 50:
            return 'High'
        elif risk_score >= 30:
            return 'Medium'
        elif risk_score > 0:
            return 'Low'
        else:
            return 'Safe'


def main():
    import sys
    
    if len(sys.argv) < 2:
        print("PHISHING WEBSITE DETECTOR WITH MACHINE LEARNING")
        print("=" * 60)
        print("\nUsage:")
        print("  Analyze URL (full):    python phishing_detector.py <URL>")
        print("  Analyze URL (ML only): python phishing_detector.py --ml-only <URL>")
        print("  Train model:           python phishing_detector.py --train <dataset.txt>")
        print("\nExamples:")
        print("  python phishing_detector.py https://example.com")
        print("  python phishing_detector.py --ml-only https://example.com")
        print("\nDataset format (one per line):")
        print("  http://example.com,0  # 0 = legitimate")
        print("  http://phish.com,1    # 1 = phishing")
        sys.exit(1)
    
    # Check for ML-only mode
    ml_only = False
    url_index = 1
    
    if sys.argv[1] == '--ml-only':
        ml_only = True
        url_index = 2
        if len(sys.argv) < 3:
            print("❌ Please provide a URL after --ml-only")
            print("Usage: python phishing_detector.py --ml-only <URL>")
            sys.exit(1)
    
    detector = PhishingDetector(use_ml=True, ml_only=ml_only)
    
    # Training mode
    if sys.argv[1] == '--train':
        if len(sys.argv) < 3:
            print("❌ Please provide dataset file path")
            print("Usage: python phishing_detector.py --train dataset.txt")
            sys.exit(1)
        
        dataset_path = sys.argv[2]
        
        # Load training data
        training_data = []
        try:
            with open(dataset_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        parts = line.split(',')
                        if len(parts) == 2:
                            url, label = parts[0].strip(), int(parts[1].strip())
                            training_data.append((url, label))
        except FileNotFoundError:
            print(f"❌ Dataset file not found: {dataset_path}")
            sys.exit(1)
        
        if not training_data:
            print("❌ No valid training data found")
            sys.exit(1)
        
        detector.train_model(training_data)
        print("\n✅ Training complete! You can now analyze URLs.")
        sys.exit(0)
    
    # Analysis mode
    url = sys.argv[url_index].strip()
    
    print("=" * 60)
    if ml_only:
        print("PHISHING DETECTION - ML ONLY MODE")
    else:
        print("PHISHING WEBSITE DETECTOR WITH MACHINE LEARNING")
    print("=" * 60)
    print(f"\n🔍 Analyzing website: {url}\n")
    
    results = detector.analyze_url(url)
    
    # Display results
    print("=" * 60)
    print(f"ANALYSIS RESULTS FOR: {results['url']}")
    print("=" * 60)
    
    if results.get('ml_prediction'):
        print(f"\n🤖 ML Prediction: {results['ml_prediction']} ({results['ml_confidence']:.1f}% confidence)")
    
    print(f"🎯 Risk Level: {results['risk_level']}")
    
    if not ml_only:
        print(f"🔒 SSL Certificate: {'Valid' if results['ssl_valid'] else 'Invalid/Missing'}")
        print(f"📝 Forms Found: {results['forms_analyzed']}")
        print(f"⚠️  Sensitive Forms: {results['sensitive_forms']}")
        print(f"✓  Has Validation: {'Yes' if results['has_validation'] else 'No'}")
    
    if results['findings']:
        print(f"\n📋 FINDINGS:")
        for finding in results['findings']:
            print(f"   {finding}")
    
    print(f"\n{'🚨 SUSPICIOUS - LIKELY PHISHING!' if results['is_suspicious'] else '✅ Appears legitimate (but stay cautious)'}")
    print("=" * 60)
    return 'PHISHING' if results['is_suspicious'] else 'LEGITIMATE'


if __name__ == "__main__":
    main()