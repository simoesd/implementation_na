// TODO delete this
// Copyright 2018 The Druid Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! Simple calculator.

// On Windows platform, don't show a console when opening the app.

use std::fs::OpenOptions;
use std::{fs::File, io::Write};

use druid::text::ParseFormatter;
use druid::{
    theme, Color, Data, Lens, RenderContext, Widget, WidgetExt,
};
use druid::widget::{CrossAxisAlignment, Flex, Label, Scroll, Painter, TextBox, Stepper, SizedBox, MainAxisAlignment};
use druid_widget_nursery::DropdownSelect;
use crate::mutationAlgorithms::{gaussianMutation::GaussianMutation, SALocalMutation::SALocalMutation, SAOnePlusOneNA::SAOnePlusOneNA, localOnePlusOneNA::LocalOnePlusOneNA, onePlusOneNA::OnePlusOneNA};
use crate::nn::ann::{NANN, self};
use crate::problems::cartpoleProblem::CartpoleProblem;
use crate::problems::naProblem::NAProblem;
use crate::problems::sphereContinuousProblem::SphereContinuousNAProblem;
use crate::problems::sphereDiscreteProblem::SphereDiscreteNAProblem;
#[derive(Clone, Data, Lens)]
pub struct AppState {
    outputFile: String,
    algorithm: AlgorithmEnum,
    problem: ProblemEnum,
    r: f64,
    numberOfPoints: f64,
    inputDim: f64,
    hiddenDim: f64,
    outputDim: f64,
    optimum: f64,
    bias: bool,
    successAdaptation: f64,
    failureAdaptation: f64,
    iterations: f64,
    results: String
}

impl AppState {
    pub fn default() -> AppState {
        AppState {
            outputFile: "".to_string(),
            algorithm: AlgorithmEnum::OnePlusOneNA,
            problem: ProblemEnum::SphereContinuous("Sphere Continuous Quarter".to_string()),
            r: 120.0,
            numberOfPoints: 1000.0,
            inputDim: 2.0,
            hiddenDim: 1.0,
            outputDim: 2.0,
            optimum: 1.0,
            bias: true,
            successAdaptation: 1.7,
            failureAdaptation: 0.9,
            results: "".to_string(),
            iterations: 1.0
        }
    }

    fn addResults(&mut self, results: String) {
        self.results.push_str(&results);
        self.results.push_str("\n");
    }

    fn clearResults(&mut self) {
        self.results = "".to_string();
    }
}

fn make_submit_button() -> impl Widget<AppState> {
    let painter = Painter::new(|ctx, _, env| {
        let bounds = ctx.size().to_rect();

        ctx.fill(bounds, &Color::rgb8(0x99, 0, 0));
        if ctx.is_hot() {
            ctx.stroke(bounds.inset(-0.5), &Color::WHITE, 1.0);
        }

        if ctx.is_active() {
            ctx.fill(bounds, &Color::rgb8(0x77, 0, 0));
        }
    });

    Label::new("Run Experiments")
        .with_text_size(18.)
        .center()
        .background(painter)
        .fix_height(40.0).fix_width(250.0)
        .on_click(move |_ctx, data: &mut AppState, _env| runUIExperiments(data)
    )
}

fn digit_button(digit: u8) -> impl Widget<AppState> {
    let painter = Painter::new(|ctx, _, env| {
        let bounds = ctx.size().to_rect();

        ctx.fill(bounds, &env.get(theme::BACKGROUND_LIGHT));

        if ctx.is_hot() {
            ctx.stroke(bounds.inset(-0.5), &Color::WHITE, 1.0);
        }

        if ctx.is_active() {
            ctx.fill(bounds, &Color::rgb8(0x71, 0x71, 0x71));
        }
    });

    Label::new(format!("Run Problems"))
        .with_text_size(24.)
        .center()
        .background(painter)
        .on_click(move |_ctx, data: &mut AppState, _env| runUIExperiments(data))
}

fn flex_row<T: Data>(
    w1: impl Widget<T> + 'static,
    w2: impl Widget<T> + 'static,
) -> impl Widget<T> {
    Flex::row()
        .with_flex_child(w1, 1.0)
        .with_spacer(1.0)
        .with_flex_child(w2, 1.0)
}

fn make_problem_row() -> Flex<AppState> {
    Flex::column().cross_axis_alignment(CrossAxisAlignment::Center).with_child(
        Flex::row().cross_axis_alignment(CrossAxisAlignment::Center).with_child(Label::new("Problem Parameters").with_text_size(18.))
    ).with_spacer(6.0).with_child(
        Flex::row().cross_axis_alignment(CrossAxisAlignment::Start).with_child(Label::new("Problem: ")).with_child(DropdownSelect::new(vec![
            ("Sphere Continuous Quarter", ProblemEnum::SphereContinuous("Sphere Continuous Quarter".to_string())),
            ("Sphere Continuous Half", ProblemEnum::SphereContinuous("Sphere Continuous Half".to_string())),
            ("Sphere Continuous Two Quarters", ProblemEnum::SphereContinuous("Sphere Continuous Two Quarters".to_string())),
            ("Sphere Continuous Local Optima", ProblemEnum::SphereContinuous("Sphere Continuous Local Optima".to_string())),
            ("Sphere Discrete 2D Quarter", ProblemEnum::SphereDiscrete("Sphere Discrete 2D Quarter".to_string())),
            ("Sphere Discrete 2D Half", ProblemEnum::SphereDiscrete("Sphere Discrete 2D Half".to_string())),
            ("Sphere Discrete 2D Two Quarters", ProblemEnum::SphereDiscrete("Sphere Discrete 2D Two Quarters".to_string())),
            ("Sphere Discrete 2D Local Optima", ProblemEnum::SphereDiscrete("Sphere Discrete 2D Local Optima".to_string())),
            ("Sphere 3D Corner", ProblemEnum::SphereDiscrete("Sphere 3D Corner".to_string())),
            ("Sphere 3D Half", ProblemEnum::SphereDiscrete("Sphere 3D Half".to_string())),
            ("Sphere 3D Slice", ProblemEnum::SphereDiscrete("Sphere 3D Slice".to_string())),
            ("Sphere 3D Two Slices", ProblemEnum::SphereDiscrete("Sphere 3D Two Slices".to_string())),
            ("Sphere 4D Quarter", ProblemEnum::SphereDiscrete("Sphere 4D Quarter".to_string())),
            ("Sphere 4D Half", ProblemEnum::SphereDiscrete("Sphere 4D Half".to_string())),
            ("Sphere 4D Two Quarters", ProblemEnum::SphereDiscrete("Sphere 4D Two Quarters".to_string())),
            ("Cartpole N Steps", ProblemEnum::Cartpole("N Steps".to_string())),
            ("Cartpole Discrete", ProblemEnum::Cartpole("Cartpole Discrete".to_string())),
        ])
        .align_left()
        .lens(AppState::problem))
        .with_spacer(16.0).with_child(Flex::row()
        .with_child(
            Label::new("Optimum: ")
        ).with_spacer(druid::theme::WIDGET_CONTROL_COMPONENT_PADDING).with_child(
            TextBox::new().with_formatter(ParseFormatter::new()).lens(AppState::optimum).fix_width(36.0)
        )
        ).with_spacer(16.0).with_child(Flex::row().with_child(
            Label::new("Number of points: ")).with_spacer(druid::theme::WIDGET_CONTROL_COMPONENT_PADDING)
            .with_child(Flex::row()
                .with_child(
                    Stepper::new()
                        .with_range(1.0, 10000.0)
                        .with_step(1.0)
                        .lens(AppState::numberOfPoints),
                )
                .with_child(
                    Label::new(|data: &f64, _env: &_| data.to_string().clone())
                    .lens(AppState::numberOfPoints)
                    .fix_width(48.0)
                )
            ).disabled_if(|appState, env| !matches!(
                appState.problem, ProblemEnum::SphereDiscrete(_) //TODO add all other discrete algorithms
                )
            )
        )
    )
}

fn make_alg_row() -> Flex<AppState> {
    Flex::column().cross_axis_alignment(CrossAxisAlignment::Center).with_child(
        Flex::row().cross_axis_alignment(CrossAxisAlignment::Center).with_child(Label::new("Algorithm Parameters").with_text_size(18.))
    ).with_spacer(6.0).with_child(
        Flex::row().main_axis_alignment(MainAxisAlignment::Center).cross_axis_alignment(CrossAxisAlignment::Center).with_child(Label::new("Mutation Algorithm: ")).with_child(DropdownSelect::new(vec![
            ("Harmonic (1+1)NA", AlgorithmEnum::OnePlusOneNA),
            ("Gaussian", AlgorithmEnum::GaussianMutation),
            ("Local (1+1)NA", AlgorithmEnum::LocalOnePlusOneNA),
            ("Self Adaptive (1+1)NA", AlgorithmEnum::SAOnePlusOneNA),
            ("Self Adaptive Local", AlgorithmEnum::SALocalMutation),
        ])
        .align_left()
        .lens(AppState::algorithm)).with_spacer(36.0)
        .with_child(Flex::column().with_child(Label::new("Self Adaptation Parameters")).with_spacer(6.0).with_child(Flex::row().with_child(
            Label::new("Success multiplier: ")).with_spacer(druid::theme::WIDGET_CONTROL_COMPONENT_PADDING)
            .with_child(Flex::row()
                .with_child(
                    Stepper::new()
                        .with_range(1.2, 2.0)
                        .with_step(0.1)
                        .lens(AppState::successAdaptation),
                ).with_child(
                    Label::new(|data: &f64, _env: &_| data.to_string().clone())
                    .lens(AppState::successAdaptation)
                    .fix_width(48.0)
                )
            ).with_child(
                Label::new("Failure multiplier: ")).with_spacer(druid::theme::WIDGET_CONTROL_COMPONENT_PADDING)
                .with_child(Flex::row()
                    .with_child(
                        Stepper::new()
                            .with_range(0.1, 1.0)
                            .with_step(0.1)
                            .lens(AppState::failureAdaptation),
                    ).with_child(
                        Label::new(|data: &f64, _env: &_| data.to_string().clone())
                        .lens(AppState::failureAdaptation)
                        .fix_width(48.0)
                    )
            ).disabled_if(|appState, env| !matches!(appState.algorithm, AlgorithmEnum::SALocalMutation) && !matches!(appState.algorithm, AlgorithmEnum::SAOnePlusOneNA)
            )
        ))
    )
}

fn make_general_row() -> Flex<AppState> {
    Flex::row().cross_axis_alignment(CrossAxisAlignment::Start).with_child(Flex::column()
    .with_child(
        Flex::row().cross_axis_alignment(CrossAxisAlignment::Center).with_child(Label::new("General Parameters").with_text_size(18.))
    ).with_spacer(6.0).with_child(Flex::column().with_child(
        Flex::row().cross_axis_alignment(CrossAxisAlignment::Center).with_child(
            Label::new("Input Dimension: ")
        ).with_spacer(druid::theme::WIDGET_CONTROL_COMPONENT_PADDING).with_child(Flex::row()
            .with_child(
                Stepper::new()
                    .with_range(1.0, 50.0)
                    .with_step(1.0)
                    .lens(AppState::inputDim),
            ).with_child(
                Label::new(|data: &f64, _env: &_| data.to_string().clone())
                .lens(AppState::inputDim)
                .fix_width(12.0)
            ).with_spacer(16.0)
        ).with_child(
            Label::new("Hidden Dimension: ")
        ).with_spacer(druid::theme::WIDGET_CONTROL_COMPONENT_PADDING).with_child(Flex::row()
            .with_child(
                Stepper::new()
                    .with_range(1.0, 50.0)
                    .with_step(1.0)
                    .lens(AppState::hiddenDim),
            ).with_child(
                Label::new(|data: &f64, _env: &_| data.to_string().clone())
                    .lens(AppState::hiddenDim)
                    .fix_width(12.0)
            ).with_spacer(16.0)
        ).with_child(
            Label::new("Output Dimension: ")
        ).with_spacer(druid::theme::WIDGET_CONTROL_COMPONENT_PADDING).with_child(Flex::row()
            .with_child(
                Stepper::new()
                    .with_range(1.0, 50.0)
                    .with_step(1.0)
                    .lens(AppState::outputDim),
            ).with_child(
                Label::new(|data: &f64, _env: &_| data.to_string().clone())
                    .lens(AppState::outputDim)
                    .fix_width(12.0)
            ).with_spacer(16.0)
        )
    ).with_spacer(8.0).with_child(Flex::row()
        .with_child(
            Label::new("Number of Iterations: ")
        ).with_spacer(druid::theme::WIDGET_CONTROL_COMPONENT_PADDING).with_child(Flex::row()
            .with_child(
                Stepper::new()
                    .with_range(1.0, 100.0)
                    .with_step(1.0)
                    .lens(AppState::iterations),
            ).with_child(
                Label::new(|data: &f64, _env: &_| data.to_string().clone())
                    .lens(AppState::iterations)
                    .fix_width(36.0)
                    .align_right()
            )
        ).with_child(
            Label::new("Resolution Parameter: ")
        ).with_spacer(druid::theme::WIDGET_CONTROL_COMPONENT_PADDING).with_child(Flex::row()
            .with_child(
                Stepper::new()
                    .with_range(1.0, 2000.0)
                    .with_step(1.0)
                    .lens(AppState::r),
            ).with_child(
                Label::new(|data: &f64, _env: &_| data.to_string().clone())
                    .lens(AppState::r)
                    .fix_width(48.0)
            )
        ).with_child(
            Label::new("Output File: ")
        ).with_spacer(druid::theme::WIDGET_CONTROL_COMPONENT_PADDING).with_child(
            TextBox::new().with_placeholder("Leave empty to not export results").fix_width(250.0).lens(AppState::outputFile)
        )
    )))
}

pub fn build_ui() -> impl Widget<AppState> {
    let mut generalRow = make_general_row();
    let problemRow = make_problem_row();
    let algRow = make_alg_row();
    let submitRow = Flex::row().main_axis_alignment(MainAxisAlignment::Center).cross_axis_alignment(CrossAxisAlignment::Center).with_child(make_submit_button());
    let resultRow = Flex::row().cross_axis_alignment(CrossAxisAlignment::Start).with_flex_child(Scroll::new(SizedBox::new(
        Label::new(|data: &String, _env: &_| data.clone())
        .with_text_size(16.0).with_text_color(Color::BLACK)
        .lens(AppState::results)
        .padding(5.0)
        .background(Color::WHITE)
    ).expand()), 1.0).must_fill_main_axis(true).expand().background(Color::WHITE);
    return Flex::column().must_fill_main_axis(true).with_spacer(24.0)
        .with_child(generalRow).with_spacer(8.0)
        .with_child(problemRow).with_spacer(8.0)
        .with_child(algRow).with_spacer(16.0)
        .with_child(submitRow).with_spacer(16.0)
        .with_flex_child(resultRow, 1.0);
}

fn runUIExperiments(data: &mut AppState) {
    AppState::clearResults(data);
    let mut problemString = "Problem: ".to_string();
    match &data.problem {
        ProblemEnum::SphereDiscrete(x) => {
            problemString.push_str(x);
            problemString.push_str(", Number of Points: ");
            problemString.push_str(&data.numberOfPoints.to_string());
        },
        ProblemEnum::SphereContinuous(x) => {
            problemString.push_str(x);
        },
        ProblemEnum::Cartpole(x) => {
            problemString.push_str(x);
        },
    }
    let mut algorithmString = "Mutation Algorithm: ".to_string();
    match &data.algorithm {
        AlgorithmEnum::SALocalMutation => {
            algorithmString.push_str("Self Adaptive Local");
            algorithmString.push_str(", Success Multiplier: ");
            algorithmString.push_str(&data.successAdaptation.to_string());
            algorithmString.push_str(", Failure Multiplier: ");
            algorithmString.push_str(&data.failureAdaptation.to_string());
        },
        AlgorithmEnum::SAOnePlusOneNA => {
            algorithmString.push_str("Self Adaptive (1+1)NA");
        },
        AlgorithmEnum::GaussianMutation => {
            algorithmString.push_str("Gaussian");
        },
        AlgorithmEnum::OnePlusOneNA => {
            algorithmString.push_str("(1+1)NA");
        },
        AlgorithmEnum::LocalOnePlusOneNA => {
            algorithmString.push_str("Local (1+1)NA");
        }
    }
    let networkString = format!("Inputs: {}, Hidden Neurons: {}, Outputs: {}, Resolution: {}, Optimum: {}\n",
        data.inputDim.to_string(),
        data.hiddenDim.to_string(),
        data.outputDim.to_string(),
        data.r.to_string(),
        data.optimum.to_string()
    );
    AppState::addResults(data, format!("Running {} experiments with parameters:", data.iterations));
    AppState::addResults(data, problemString.to_string());
    AppState::addResults(data, algorithmString.to_string());
    AppState::addResults(data, networkString);
    
    unsafe { crate::R = data.r };
    unsafe { crate::OPTIMUM = data.optimum };
    if data.outputFile != "" {
        let mut resultFile = File::create(&data.outputFile).expect("Unable to created final result file");
        writeln!(resultFile, "R,Mutation Algorithm,Problem,Input Dim,Hidden Dim,Output Dim,Iteration,Generation,Score,Solution").expect("Failed writing result file");
    }
    for i in 1..data.iterations as i32 + 1 {
        // [(inputs, number of hidden neurons) (number of hidden neurons, number of hidden neurons), (number of hidden neurons, outputs)]
        let nn = NANN::new(vec![(data.inputDim as usize, data.hiddenDim as usize), (data.hiddenDim as usize, data.outputDim as usize)], |x| x, data.r, true);
        let problem: Box<dyn NAProblem> = match &data.problem {
            ProblemEnum::SphereDiscrete(x) => {
                match x.as_str() {
                    "Sphere Discrete 2D Quarter" => SphereDiscreteNAProblem::newQuarter(data.numberOfPoints as u32),
                    "Sphere Discrete 2D Half" => SphereDiscreteNAProblem::newHalf(data.numberOfPoints as u32),
                    "Sphere Discrete 2D Two Quarters" => SphereDiscreteNAProblem::newTwoQuarters(data.numberOfPoints as u32),
                    "Sphere Discrete 2D Local Optima" => SphereDiscreteNAProblem::newLocalOpt(data.numberOfPoints as u32),
                    "Sphere 3D Corner" => SphereDiscreteNAProblem::newCorner3D(data.numberOfPoints as u32),
                    "Sphere 3D Half" => SphereDiscreteNAProblem::newHalf3D(data.numberOfPoints as u32),
                    "Sphere 3D Slice" => SphereDiscreteNAProblem::newSlice3D(data.numberOfPoints as u32),
                    "Sphere 3D Two Slices" => SphereDiscreteNAProblem::newTwoSlices3D(data.numberOfPoints as u32),
                    "Sphere 4D Quarter" => SphereDiscreteNAProblem::newQuarter4D(data.numberOfPoints as u32),
                    "Sphere 4D Half" => SphereDiscreteNAProblem::newHalf4D(data.numberOfPoints as u32),
                    "Sphere 4D Two Quarters" => SphereDiscreteNAProblem::newTwoQuarters4D(data.numberOfPoints as u32),
                    _ => SphereDiscreteNAProblem::newQuarter(data.numberOfPoints as u32),
                }
            },
            ProblemEnum::SphereContinuous(x) => {
                match x.as_str() {
                    "Sphere Continuous Quarter" => SphereContinuousNAProblem::newQuarter(),
                    "Sphere Continuous Half" => SphereContinuousNAProblem::newHalf(),
                    "Sphere Continuous Two Quarters" => SphereContinuousNAProblem::newTwoQuarters(),
                    "Sphere Continuous Local Optima" => SphereContinuousNAProblem::newLocalOpt(),
                    _ => SphereContinuousNAProblem::newQuarter()
                }
            },
            ProblemEnum::Cartpole(x) => {
                match x.as_str() {
                    "Cartpole N Steps" => CartpoleProblem::newNSteps(),
                    "Cartpole Discrete" => CartpoleProblem::newDiscretizedContinuous(),
                    _ => CartpoleProblem::newNSteps(),
                }
            },
        };
        let mutationAlgorithm = match &data.algorithm {
            AlgorithmEnum::SALocalMutation => {
                SALocalMutation::new(&nn, problem, data.r, data.successAdaptation, data.failureAdaptation)
            },
            AlgorithmEnum::SAOnePlusOneNA => {
                SAOnePlusOneNA::new(&nn, problem, data.r, data.successAdaptation, data.failureAdaptation)
            },
            AlgorithmEnum::GaussianMutation => {
                GaussianMutation::new(problem)
            },
            AlgorithmEnum::OnePlusOneNA => {
                OnePlusOneNA::new(&nn, problem, data.r)
            },
            AlgorithmEnum::LocalOnePlusOneNA => {
                LocalOnePlusOneNA::new(&nn, problem, data.r)
            }
        };
            
        let problemName = mutationAlgorithm.getProblem().to_string();
        let mutationAlgorithmName = mutationAlgorithm.to_string();
        let (generation, _, score, solution) = ann::run(
            nn,
            mutationAlgorithm
        );
        
        let solutionString: String = solution.map(|x| ((x*1000.0).round()/1000.0).to_string()).into_raw_vec().join("; ");
        AppState::addResults(data, format!("Iteration {i}/{}: finished in generation {generation} with a score of {}", data.iterations, (score*1000.0).round()/1000.0));
        AppState::addResults(data, format!("Solution found: [{solutionString}]"));
        if data.outputFile != "" {
            let mut resultFile = OpenOptions::new().append(true).open(&data.outputFile).unwrap();
            let exportSolutionString: String = solution.map(|x| x.to_string()).into_raw_vec().join(";");
            writeln!(resultFile, "{},{},{},{},{},{},{},{},{},{}",
                data.r, mutationAlgorithmName, problemName, data.inputDim, data.hiddenDim, data.outputDim, i, generation, score, exportSolutionString
            ).expect("Failed writing result file");
        }
        };
}

#[derive(Data, Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Debug)]
enum AlgorithmEnum {
    LocalOnePlusOneNA,
    GaussianMutation,
    OnePlusOneNA,
    SALocalMutation,
    SAOnePlusOneNA
}
#[derive(Data, Clone, Eq, PartialEq, Ord, PartialOrd, Debug)]
enum ProblemEnum {
    SphereDiscrete(String),
    SphereContinuous(String),
    Cartpole(String),
}