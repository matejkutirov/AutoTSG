import argparse
from src.pipeline import run_pipeline
from src.automl import automl_model_selection
from config.config import MODEL_SELECTION_CONFIG


def main():
    parser = argparse.ArgumentParser(
        description="Time Series Generation and AutoML Pipeline"
    )
    parser.add_argument(
        "--ds",
        "--dataset",
        required=True,
        help="Dataset name (google, air, appliances, eeg) or file path",
    )
    parser.add_argument(
        "--model",
        "--models",
        required=True,
        nargs="+",  # Accept one or more model names
        help="Model type(s): single model (e.g., arima), multiple models (e.g., timevae timegan arima), or 'all' for all available models",
    )
    parser.add_argument(
        "--ms",
        "--model_selection",
        help="Model selection method: 'automl' for FLAML-based selection",
    )
    parser.add_argument("--file", help="File path for custom dataset")
    parser.add_argument(
        "--time_budget",
        type=int,
        default=300,
        help="Time budget in seconds for AutoML (default: 300)",
    )
    parser.add_argument(
        "--metric",
        choices=[
            "rmse",
            "mae",
            "ed",
            "dtw",
            "mdd",
            "acd",
            "sd",
            "kd",
            "ds",
            "ps",
            "c-fid",
            "RMSE",
            "MAE",
            "ED",
            "DTW",
            "MDD",
            "ACD",
            "SD",
            "KD",
            "DS",
            "PS",
            "C-FID",
        ],
        default="MAE",
        help="Optimization metric for AutoML model selection. Available metrics: 'rmse' (Root Mean Square Error), 'mae' (Mean Absolute Error), 'ed' (Euclidean Distance), 'dtw' (Dynamic Time Warping), 'mdd' (Marginal Distribution Difference), 'acd' (Autocorrelation Difference), 'sd' (Skewness Difference), 'kd' (Kurtosis Difference), 'ds' (Discriminative Score), 'ps' (Predictive Score), 'c-fid' (Contextual-FID). Case-insensitive. Default: MAE",
    )
    parser.add_argument(
        "--list-metrics",
        action="store_true",
        help="List all available evaluation metrics with descriptions",
    )

    args = parser.parse_args()

    # Handle list-metrics argument
    if args.list_metrics:
        from src.automl import list_available_metrics

        list_available_metrics()
        return

    if args.ds.endswith((".csv", ".xlsx", ".xls")):
        file_path = args.ds
        dataset = "custom"
    else:
        dataset = args.ds
        file_path = args.file

    # Check if using AutoML
    if args.ms == "automl":
        print(f"Running AutoML model selection for {dataset}")
        print(f"Models to test: {', '.join(args.model)}")
        print(f"Time budget: {args.time_budget} seconds")
        print("=" * 50)

        try:
            # Run AutoML with specified models
            results = automl_model_selection(
                dataset, args.time_budget, metric=args.metric, models_to_test=args.model
            )

            print(f"\nAutoML completed successfully!")
            print(f"Best model: {results['best_model_type']}")
            print(f"Best score: {results['best_score']:.4f}")

            # Print search summary
            if "automl_results" in results:
                print(f"\nSearch Summary:")
                print(f"  ✓ Best model: {results['best_model_type']}")
                print(f"  ✓ Best score: {results['best_score']:.4f}")
                print(f"  ✓ Best config: {results['best_config']}")

        except Exception as e:
            print(f"AutoML failed: {e}")
            import traceback

            traceback.print_exc()

    else:
        # Run standard pipeline
        # Handle different model scenarios
        if args.model == ["all"]:
            # Run all available models
            all_models = MODEL_SELECTION_CONFIG["available_models"]
            print(f"Running all available models: {', '.join(all_models)}")

            all_results = []
            for model_name in all_models:
                try:
                    print(f"\n{'=' * 50}")
                    print(f"Training model: {model_name}")
                    print(f"{'=' * 50}")

                    if dataset == "air":
                        model_results, metrics, generated_data, val_sequences = (
                            run_pipeline(dataset, model_name, file_path)
                        )
                        all_results.append(
                            (
                                model_name,
                                model_results,
                                metrics,
                                generated_data,
                                val_sequences,
                            )
                        )
                    else:
                        model_results, metrics, generated_data = run_pipeline(
                            dataset, model_name, file_path
                        )
                        all_results.append(
                            (model_name, model_results, metrics, generated_data)
                        )

                except Exception as e:
                    print(f"Error training {model_name}: {e}")
                    continue

            # Print summary of all results
            print(f"\n{'=' * 50}")
            print("PIPELINE SUMMARY - ALL MODELS")
            print(f"{'=' * 50}")
            for model_name, model_results, metrics, *rest in all_results:
                print(
                    f"✓ {model_name}: {model_results['model_type']} - Training time: {metrics.get('Training_Time', 'N/A'):.2f}s"
                )

            # Return results for the last successful model (for compatibility)
            if all_results:
                last_result = all_results[-1]
                if dataset == "air":
                    return (
                        last_result[1],
                        last_result[2],
                        last_result[3],
                        last_result[4],
                    )
                else:
                    return last_result[1], last_result[2], last_result[3]
            else:
                print("No models were successfully trained.")
                return None, None, None

        elif len(args.model) > 1:
            # Run multiple specified models
            print(f"Running multiple models: {', '.join(args.model)}")

            all_results = []
            for model_name in args.model:
                try:
                    print(f"\n{'=' * 50}")
                    print(f"Training model: {model_name}")
                    print(f"{'=' * 50}")

                    if dataset == "air":
                        model_results, metrics, generated_data, val_sequences = (
                            run_pipeline(dataset, model_name, file_path)
                        )
                        all_results.append(
                            (
                                model_name,
                                model_results,
                                metrics,
                                generated_data,
                                val_sequences,
                            )
                        )
                    else:
                        model_results, metrics, generated_data = run_pipeline(
                            dataset, model_name, file_path
                        )
                        all_results.append(
                            (model_name, model_results, metrics, generated_data)
                        )

                except Exception as e:
                    print(f"Error training {model_name}: {e}")
                    continue

            # Print summary of all results
            print(f"\n{'=' * 50}")
            print("PIPELINE SUMMARY - MULTIPLE MODELS")
            print(f"{'=' * 50}")
            for model_name, model_results, metrics, *rest in all_results:
                print(
                    f"✓ {model_name}: {model_results['model_type']} - Training time: {metrics.get('Training_Time', 'N/A'):.2f}s"
                )

            # Return results for the last successful model (for compatibility)
            if all_results:
                last_result = all_results[-1]
                if dataset == "air":
                    return (
                        last_result[1],
                        last_result[2],
                        last_result[3],
                        last_result[4],
                    )
                else:
                    return last_result[1], last_result[2], last_result[3]
            else:
                print("No models were successfully trained.")
                return None, None, None

        else:
            # Run single model
            model_name = args.model[0]
            print(f"Running single model: {model_name}")

            if dataset == "air":
                # Air dataset returns 4 values (domain adaptation)
                model_results, metrics, generated_data, val_sequences = run_pipeline(
                    dataset, model_name, file_path
                )
                print(f"\nPipeline completed successfully!")
                print(f"Dataset: {dataset} (Domain Adaptation)")
                print(f"Model: {model_name}")
                print(f"Source Domain (Tianjin): Training + Validation")
                print(f"Target Domains (BJ/GZ/SZ): Testing")
            else:
                # Other datasets returns 3 values
                model_results, metrics, generated_data = run_pipeline(
                    dataset, model_name, file_path
                )
                print(f"\nPipeline completed successfully!")
                print(f"Dataset: {dataset}")
                print(f"Model: {model_name}")


if __name__ == "__main__":
    main()
