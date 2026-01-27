# Frontend Note

## cache

* 除了 echarts，前端还有哪些类似 matplotlib 的可视化库

    除了 ECharts，前端还有不少优秀的可视化库，以下是一些类似 Matplotlib 风格（注重数据驱动、图表类型丰富）的库：

    1. 基于 SVG/Canvas 的通用库

        D3.js

        * 特点：底层数据驱动库，高度灵活，可定制任何可视化效果。

        * 适用场景：复杂定制化图表、数据可视化研究。

        * 示例：力导向图、地图定制、动态交互。

        * 官网：d3js.org

            <https://d3js.org/>

        Chart.js

        * 特点：轻量级（约 60KB）、简单易用，支持 Canvas 渲染。

        * 适用场景：移动端友好、基础图表（折线、柱状、饼图等）。

        * 官网：chartjs.org

            <https://www.chartjs.org/>

        Highcharts

        * 特点：商业级图表库，兼容性好，支持导出功能。

        * 适用场景：企业报表、商业数据分析。

        * 许可证：免费用于非商业项目，商业需付费。

        * 官网：highcharts.com

            <https://www.highcharts.com/>

        Plotly.js

        * 特点：基于 D3.js 和 WebGL，支持 3D 图表、科学图表。

        * 适用场景：交互式科学图表、金融数据可视化。

        * 官网：plotly.com/javascript

            <https://plot.ly/javascript/>

    2. 声明式语法库

        Vega / Vega-Lite

        * 特点：JSON 语法声明图表，无需编程即可生成可视化。

        * 适用场景：配置化图表生成、学术可视化。

        * 官网：vega.github.io

            <https://vega.github.io/>

    3. 面向框架的库

        Recharts（React）

        * 特点：基于 React 和 D3，组件化声明式语法。

        * 适用场景：React 项目中的简单图表集成。

        * 官网：recharts.org

            <https://recharts.org/>

        Victory（React/React Native）

        * 特点：React 生态中的模块化图表库，支持移动端。

        * 适用场景：React Native 或 React 项目。

        * 官网：formidable.com/open-source/victory

            <https://formidable.com/open-source/victory/>

        ApexCharts

        * 特点：现代交互式图表，支持 Vue/React/React Native。

        * 适用场景：响应式仪表盘、实时数据更新。

        * 官网：apexcharts.com

            <https://apexcharts.com/>

    4. 专业领域库

        Three.js / Babylon.js

        * 特点：WebGL 3D 渲染库，适合复杂 3D 可视化。

        * 适用场景：3D 模型、地理可视化、游戏化数据展示。

        * 官网：threejs.org / babylonjs.com

            <https://threejs.org/>

            <https://www.babylonjs.com/>

        Mapbox GL JS / Leaflet

        * 特点：专业地图可视化库。

        * 适用场景：地理数据、轨迹热力图等。

        * 官网：mapbox.com / leafletjs.com

            <https://docs.mapbox.com/mapbox-gl-js/>

            <https://leafletjs.com/>

    5. 轻量级或新兴库

        AntV 系列（蚂蚁集团）

        * 特点：包含 G2（通用图表）、G6（图可视化）、L7（地理可视化）。

        * 适用场景：企业级复杂可视化需求。

        * 官网：antv.vision

            <https://antv.vision/>

        Observable Plot

        * 特点：D3 作者推出的轻量级库，语法简洁。

        * 适用场景：快速生成基础图表、数据探索。

        * 官网：observablehq.com/plot

            <https://observablehq.com/plot/>

        Chartist.js

        * 特点：响应式、CSS 驱动动画的轻量库。

        * 适用场景：移动端简单图表。

        * 官网：gionkunz.github.io/chartist-js

            <https://observablehq.com/plot/>

    选择建议

    * 快速上手：Chart.js、Recharts（React）、ApexCharts。

    * 高度定制：D3.js、Three.js。

    * 商业报表：Highcharts、ECharts。

    * 学术/科学可视化：Plotly.js、Vega-Lite。

    * 地理数据：Mapbox GL JS、Leaflet。

    这些库各有侧重，可根据项目需求（如性能、兼容性、交互复杂度）选择。

## topics
