from sklearn.feature_extraction.text import TfidfVectorizer
import os
import joblib
import pandas as pd
import numpy as np

class Preprocessor:
    """Preprocessor that contains a sequence of procedures"""

    async def drop_constant_columns(self, datamanager):
        """Inspect the data quality report for constant columns and drop them"""
        if not hasattr(datamanager, "df") or datamanager.df is None:
            raise RuntimeError("\ndatamanager has no loaded DataFrame (df is None)")

        data_quality = await datamanager.get_data_quality_report()
        const_cols = data_quality.get("constant_columns", [])

        if not const_cols:
            print("\nNo constant columns detected.")
            return []

        datamanager.df = datamanager.df.drop(columns=const_cols)
        print(f"\nDropped constant columns: {const_cols}")


    async def handle_missing_values(self, datamanager, threshold: float = 50.0):
        """
        Inspect data quality report and:
         - drop columns with completeness_rate < threshold
         - for remaining columns with missing values, impute selected text columns with "missing"
        """
        if not hasattr(datamanager, "df") or datamanager.df is None:
            raise RuntimeError("\ndatamanager has no loaded DataFrame (df is None)")

        # obtain report
        data_quality = await datamanager.get_data_quality_report()
        
        missing_data_cols = data_quality.get('completeness', {})
        # Determine columns to drop based on high missing percentage
        cols_to_drop = [col for col, stats in missing_data_cols.items()
                        if stats.get('completeness_rate', 100) < threshold]

        if cols_to_drop:
            print(f"\nDropping columns with high missing data: {cols_to_drop}")
            datamanager.df = datamanager.df.drop(columns=cols_to_drop)

        # Handle missing values in remaining columns
        remaining_missing_cols = {col: stats for col, stats in missing_data_cols.items()
                                  if col in datamanager.df.columns and stats.get('missing_count', 0) > 0}

        for col, stats in remaining_missing_cols.items():
            missing_percentage = 100.0 - float(stats.get('completeness_rate', 100))
            print(f"\nHandling missing values in '{col}' ({missing_percentage:.1f}% missing)")

            if col in ['Subject', 'X-To', 'X-FileName']:
                datamanager.df[col] = datamanager.df[col].fillna("missing")
                print(f"\nImputed missing values in '{col}' with 'missing'")
    
    async def create_text_features(self, datamanager, top_k_domains: int = 50):
        """
        Adds to datamanager.df:
          - text_combined: Subject + ' ' + Body
          - text_length: int length
          - contains_urls, contains_email_addresses, contains_phone_numbers, contains_money_symbols (0/1 ints)
          - sender_domain, recipient_domain (strings)
        Encodes sender_domain and recipient_domain as one-hot columns and sets
        datamanager.encoded_feature_columns to the list of created feature names.
        """
        if not hasattr(datamanager, "df") or datamanager.df is None:
            raise RuntimeError("datamanager has no loaded DataFrame (df is None)")

        df = datamanager.df

        # Safely combine Subject and Body
        subj = df.get('Subject', pd.Series([], dtype=str)).fillna("").astype(str)
        body = df.get('Body', pd.Series([], dtype=str)).fillna("").astype(str)
        df['text_combined'] = (subj + ' ' + body).str.strip()

        # Basic text-derived features
        df['text_length'] = df['text_combined'].str.len().fillna(0).astype(int)
        df['contains_urls'] = df['text_combined'].str.contains(r'https?://|www\.', case=False, na=False).astype(int)
        df['contains_email_addresses'] = df['text_combined'].str.contains(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", na=False).astype(int)
        df['contains_phone_numbers'] = df['text_combined'].str.contains(r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b", na=False).astype(int)
        df['contains_money_symbols'] = df['text_combined'].str.contains(r"[\$£€¥]|\bmoney\b|\bcash\b|\bpayment\b", case=False, na=False).astype(int)
        df['contains_mentions'] = df['text_combined'].str.contains(r"\battach|\bdocument\b|\bfile\b|\bdownload\b", case=False, na=False).astype(int)
        
        # Extract domains
        if 'X-From' in df.columns:
            df['sender_domain'] = df['X-From'].astype(str).str.extract(r"@([^@\s,<>]+)", expand=False).fillna('unknown')
        else:
            df['sender_domain'] = 'unknown'

        if 'X-To' in df.columns:
            df['recipient_domain'] = df['X-To'].astype(str).str.extract(r"@([^@\s,<>]+)", expand=False).fillna('unknown')
        else:
            df['recipient_domain'] = 'unknown'

        print("New feature columns 'sender_domain' and 'recipient_domain' created.")
        print(df[['sender_domain', 'recipient_domain']].head())

        # One-hot encode only the top-k domains to avoid explosion in column count
        try:
            top_k = int(top_k_domains) if top_k_domains is not None else 50
        except Exception:
            top_k = 50

        sender_top = df['sender_domain'].value_counts().nlargest(top_k).index.tolist()
        recipient_top = df['recipient_domain'].value_counts().nlargest(top_k).index.tolist()

        df['sender_domain_trunc'] = df['sender_domain'].where(df['sender_domain'].isin(sender_top), other='other')
        df['recipient_domain_trunc'] = df['recipient_domain'].where(df['recipient_domain'].isin(recipient_top), other='other')

        try:
            sender_dummies = pd.get_dummies(df['sender_domain_trunc'], prefix='sender_dom', dtype=np.uint8)
            recipient_dummies = pd.get_dummies(df['recipient_domain_trunc'], prefix='recipient_dom', dtype=np.uint8)

            # Concat encoded columns
            df = pd.concat([df, sender_dummies, recipient_dummies], axis=1)
            datamanager.df = df

            # Collect feature column names
            encoded_cols = list(sender_dummies.columns) + list(recipient_dummies.columns)
        except MemoryError:
            # Fallback: label encode to avoid huge memory use
            print("MemoryError while one-hot encoding domains; falling back to label encoding for domains.")
            df['sender_domain_label'] = pd.factorize(df['sender_domain'])[0].astype(np.int32)
            df['recipient_domain_label'] = pd.factorize(df['recipient_domain'])[0].astype(np.int32)
            datamanager.df = df
            encoded_cols = ['sender_domain_label', 'recipient_domain_label']
        feature_cols = ['text_length', 'contains_urls', 'contains_email_addresses', 'contains_phone_numbers', 'contains_money_symbols'] + encoded_cols
        setattr(datamanager, 'encoded_feature_columns', feature_cols)

        print(f"Created {len(feature_cols)} ML feature columns (stored in datamanager.encoded_feature_columns)")
        return feature_cols
    
    async def vectorize_text(self, datamanager, 
                             text_columns: list | None = None,
                             vectorizer_type: str = "tfidf",
                             ngram_range: tuple = (1, 2),
                             max_features: int = 10000):
        """
        Convert text columns into numeric vectors suitable for ML.
        Either fits and stores a new vectorizer or load one if already computed.
        Returns the feature matrix on success.
        """
        if not hasattr(datamanager, "df") or datamanager.df is None:
            raise RuntimeError("datamanager has no loaded DataFrame (df is None)")

        # default text columns: prefer combined text if available
        if text_columns is None:
            if 'text_combined' in datamanager.df.columns:
                text_columns = ['text_combined']
            else:
                text_columns = ["Subject", "Body"]

        # Combine selected text columns into a single text series
        texts = datamanager.df[text_columns[0]].fillna("").astype(str)
        for col in text_columns[1:]:
            texts = texts + " " + datamanager.df[col].fillna("").astype(str)

        # create cache folder path under app/vectorizers
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'vectorizers'))
        os.makedirs(base_dir, exist_ok=True)

        # build filename from parameters
        cols_key = '+'.join(text_columns)
        ngram_key = f"{ngram_range[0]}-{ngram_range[1]}"
        fname = f"vectorizer__type-{vectorizer_type}__cols-{cols_key}__ngram-{ngram_key}__maxf-{max_features}.joblib"
        fpath = os.path.join(base_dir, fname)

        # If vectorizer already exists with the same parameters, load and transform
        if os.path.exists(fpath):
            try:
                vectorizer = joblib.load(fpath)
                X = vectorizer.transform(texts)
                setattr(datamanager, "X_processed", X)
                setattr(datamanager, "vectorizer", vectorizer)
                setattr(datamanager, "text_columns_used", text_columns)
                print(f"Loaded vectorizer from cache: {fpath}. Feature matrix shape: {X.shape}")
                return X
            except Exception as e:
                print(f"Failed to load cached vectorizer ({fpath}), will refit. Error: {e}")

        # create and fit vectorizer
        if vectorizer_type.lower() != "tfidf":
            raise ValueError(f"Unsupported vectorizer_type: {vectorizer_type}")

        vectorizer = TfidfVectorizer(lowercase=True, strip_accents="unicode",
                                     ngram_range=ngram_range, max_features=max_features)

        X = vectorizer.fit_transform(texts)

        # store results on datamanager for downstream use
        setattr(datamanager, "X_processed", X)
        setattr(datamanager, "vectorizer", vectorizer)
        setattr(datamanager, "text_columns_used", text_columns)

        # persist vectorizer to cache for future runs
        try:
            joblib.dump(vectorizer, fpath)
            print(f"Saved vectorizer to: {fpath}")
        except Exception as e:
            print(f"Warning: failed to save vectorizer to {fpath}: {e}")

        print(f"Vectorization complete. Feature matrix shape: {X.shape}")
        return X
    


